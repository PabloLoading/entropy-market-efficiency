"""H2 enhanced: Random Forest con features extras (volatility pre-peak, reversal).

Setup:
- Modelo A (baseline sin PE): dd_actual, vol_realizada, vol_pre_peak, reversal_5d
- Modelo B (con PE): agrega pe_t

Features:
- pe_t                : PE al día de entrada al evento (dd cruza -5%)
- drawdown_actual_t   : cuánto cayó ya el precio desde el peak
- vol_realizada_t     : std de log-retornos desde peak hasta t
- vol_pre_peak        : std de log-retornos 60 días antes del peak
- reversal_5d_t       : retorno acumulado de los últimos 5 días previos a t

Evaluación OOS: R^2 test, MAE test, feature importance de Random Forest.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from helpers import (
    ASSETS_H2, MIN_EVENT_DAYS_H2, OUTPUTS, detect_events, load_price_ret_pe,
)
from run_h2 import CONFIGS, ENTRY_DD, EXPERIMENTS

VOL_PRE_WINDOW = 60
REVERSAL_WINDOW = 5
RF_N_ESTIMATORS = 200
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_DEPTH = 6
RF_RANDOM_STATE = 42


def build_enhanced_obs(events, close, log_ret, pe, ticker,
                         min_event_days=MIN_EVENT_DAYS_H2):
    rows = []
    for i, ev in events.iterrows():
        peak, trough = ev["peak_date"], ev["trough_date"]
        if (trough - peak).days < min_event_days:
            continue
        during = pe.loc[peak:trough]
        if len(during) < 5:
            continue
        peak_price = ev["peak_price"]

        vol_before = log_ret.loc[:peak]
        if len(vol_before) < VOL_PRE_WINDOW:
            continue
        vol_pre_peak = float(vol_before.iloc[-VOL_PRE_WINDOW:].std())

        entry_t = None
        for t in during.index:
            price_t = close.loc[t] if t in close.index else np.nan
            if not np.isfinite(price_t):
                continue
            if (price_t - peak_price) / peak_price <= ENTRY_DD:
                entry_t = t
                break
        if entry_t is None:
            continue

        pe_t = pe.loc[entry_t]
        if not np.isfinite(pe_t):
            continue

        ret_since_peak = log_ret.loc[peak:entry_t]
        if len(ret_since_peak) < 5:
            continue
        vol_realizada = float(ret_since_peak.std())

        ret_before_entry = log_ret.loc[:entry_t]
        if len(ret_before_entry) < REVERSAL_WINDOW:
            continue
        reversal_5d = float(ret_before_entry.iloc[-REVERSAL_WINDOW:].sum())

        rows.append({
            "event_id": f"{ticker}_{int(i)}",
            "ticker": ticker,
            "peak_date": peak,
            "entry_date": entry_t,
            "pe_t": float(pe_t),
            "drawdown_actual_t": (close.loc[entry_t] - peak_price) / peak_price,
            "vol_realizada_t": vol_realizada,
            "vol_pre_peak": vol_pre_peak,
            "reversal_5d_t": reversal_5d,
            "trough_final": abs(ev["drawdown"]),
            "event_max_dd": abs(ev["drawdown"]),
        })
    return pd.DataFrame(rows)


def build_multi_asset_enhanced(pe_window, m, assets):
    dfs = []
    for asset in assets:
        close, log_ret, pe = load_price_ret_pe(
            pe_window=pe_window, m=m,
            ticker=asset["ticker"], start=asset["start"],
        )
        events = detect_events(close)
        df = build_enhanced_obs(events, close, log_ret, pe, ticker=asset["ticker"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def r2(y, pred):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_rf(X, y):
    return RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    ).fit(X, y)


def run_config(cfg, experiment):
    df = build_multi_asset_enhanced(pe_window=cfg["W"], m=3, assets=experiment["assets"])
    df = df.dropna(subset=["pe_t", "drawdown_actual_t", "vol_realizada_t",
                             "vol_pre_peak", "reversal_5d_t"])
    if cfg["cutoff"] is not None:
        df = df[df["event_max_dd"] <= cfg["cutoff"]].copy()

    train = df[df["peak_date"] < experiment["train_end"]].copy()
    test = df[df["peak_date"] >= experiment["test_start"]].copy()

    cols_a = ["drawdown_actual_t", "vol_realizada_t", "vol_pre_peak", "reversal_5d_t"]
    cols_b = ["pe_t"] + cols_a

    y_tr = train["trough_final"].values
    m_a = fit_rf(train[cols_a].values, y_tr)
    m_b = fit_rf(train[cols_b].values, y_tr)

    y_te = test["trough_final"].values
    pred_a = m_a.predict(test[cols_a].values)
    pred_b = m_b.predict(test[cols_b].values)

    fi_b = dict(zip(cols_b, m_b.feature_importances_.tolist()))

    return {
        "config": cfg["label"],
        "train_n": len(train),
        "test_n": len(test),
        "r2_test_a": r2(y_te, pred_a),
        "r2_test_b": r2(y_te, pred_b),
        "delta_r2_test": r2(y_te, pred_b) - r2(y_te, pred_a),
        "mae_test_a": float(np.mean(np.abs(y_te - pred_a))),
        "mae_test_b": float(np.mean(np.abs(y_te - pred_b))),
        "feat_importance_with_pe": {k: round(v, 4) for k, v in fi_b.items()},
    }


def main():
    all_results = {}
    for exp in EXPERIMENTS:
        results = [run_config(c, exp) for c in CONFIGS]
        all_results[exp["name"]] = {"label": exp["label"], "results": results}
        print(f"\n=== H2 ENHANCED (Random Forest) [{exp['label']}] ===\n")
        print(f"{'config':16s} {'tr_n':>5s} {'te_n':>5s} "
              f"{'R2_A_te':>8s} {'R2_B_te':>8s} {'ΔR2':>9s} {'MAE_A':>7s} {'MAE_B':>7s}")
        for r in results:
            print(f"{r['config']:16s} {r['train_n']:>5d} {r['test_n']:>5d} "
                  f"{r['r2_test_a']:>+8.3f} {r['r2_test_b']:>+8.3f} "
                  f"{r['delta_r2_test']:>+9.4f} {r['mae_test_a']:>7.4f} "
                  f"{r['mae_test_b']:>7.4f}")
        print("\nFeature importance (mejor config, modelo con PE):")
        best = max(results, key=lambda x: x["r2_test_b"])
        for feat, imp in sorted(best["feat_importance_with_pe"].items(),
                                  key=lambda x: -x[1]):
            print(f"  {feat:22s} {imp:.3f}")

    with open(OUTPUTS / "e3_h2_ml_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
