"""H2: ¿PE_t al inicio del evento aporta información sobre el trough final?

Setup limpio: una observación por evento (primer día con drawdown <= -5%).
Regresión OLS: trough_final ~ PE_t + drawdown_actual_t + vol_realizada_t.
Comparación: modelo A (sin PE) vs modelo B (con PE).
Split temporal: train peak<2015, embargo 2 años, test peak>=2017.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from helpers import (
    ASSETS_H2, MIN_EVENT_DAYS_H2, OUTPUTS, detect_events, load_price_ret_pe,
)

CONFIGS = [
    {"label": "W=140_all", "W": 140, "cutoff": None},
    {"label": "W=140_leq25", "W": 140, "cutoff": 0.25},
    {"label": "W=180_all", "W": 180, "cutoff": None},
    {"label": "W=180_leq25", "W": 180, "cutoff": 0.25},
]

EXPERIMENTS = [
    {
        "name": "multi_asset_new",
        "label": "14 activos, split 2015/2017",
        "assets": ASSETS_H2,
        "train_end": pd.Timestamp("2015-01-01"),
        "test_start": pd.Timestamp("2017-01-01"),
    },
    {
        "name": "gspc_only_old",
        "label": "solo ^GSPC, split 2000/2002",
        "assets": [{"ticker": "^GSPC", "start": "1928-01-01"}],
        "train_end": pd.Timestamp("2000-01-01"),
        "test_start": pd.Timestamp("2002-01-01"),
    },
]

ENTRY_DD = -0.05


def build_early_obs(events, close, log_ret, pe, ticker,
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
        entry_t = None
        for t in during.index:
            price_t = close.loc[t] if t in close.index else np.nan
            if not np.isfinite(price_t):
                continue
            dd_actual = (price_t - peak_price) / peak_price
            if dd_actual <= ENTRY_DD:
                entry_t = t
                break
        if entry_t is None:
            continue
        pe_t = pe.loc[entry_t]
        if not np.isfinite(pe_t):
            continue
        ret_window = log_ret.loc[peak:entry_t]
        if len(ret_window) < 5:
            continue
        rows.append({
            "event_id": f"{ticker}_{int(i)}",
            "ticker": ticker,
            "peak_date": peak,
            "entry_date": entry_t,
            "pe_t": float(pe_t),
            "drawdown_actual_t": (close.loc[entry_t] - peak_price) / peak_price,
            "vol_realizada_t": float(ret_window.std()),
            "trough_final": abs(ev["drawdown"]),
            "event_max_dd": abs(ev["drawdown"]),
        })
    return pd.DataFrame(rows)


def build_multi_asset_early(pe_window, m, assets):
    dfs = []
    for asset in assets:
        close, log_ret, pe = load_price_ret_pe(
            pe_window=pe_window, m=m,
            ticker=asset["ticker"], start=asset["start"],
        )
        events = detect_events(close)
        df = build_early_obs(events, close, log_ret, pe, ticker=asset["ticker"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def r2(y, pred):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def run_config(cfg, experiment):
    df = build_multi_asset_early(pe_window=cfg["W"], m=3, assets=experiment["assets"])
    df = df.dropna(subset=["pe_t", "drawdown_actual_t", "vol_realizada_t"])
    if cfg["cutoff"] is not None:
        df = df[df["event_max_dd"] <= cfg["cutoff"]].copy()

    train = df[df["peak_date"] < experiment["train_end"]].copy()
    test = df[df["peak_date"] >= experiment["test_start"]].copy()

    cols_a = ["drawdown_actual_t", "vol_realizada_t"]
    cols_b = ["pe_t", "drawdown_actual_t", "vol_realizada_t"]

    y_tr = train["trough_final"]
    m_a = sm.OLS(y_tr, sm.add_constant(train[cols_a])).fit()
    m_b = sm.OLS(y_tr, sm.add_constant(train[cols_b])).fit()

    y_te = test["trough_final"].values
    pred_a = m_a.predict(sm.add_constant(test[cols_a]))
    pred_b = m_b.predict(sm.add_constant(test[cols_b]))
    mae_a = float(np.mean(np.abs(y_te - pred_a)))
    mae_b = float(np.mean(np.abs(y_te - pred_b)))

    return {
        "config": cfg["label"],
        "train_n": len(train),
        "test_n": len(test),
        "beta_pe": float(m_b.params["pe_t"]),
        "p_pe": float(m_b.pvalues["pe_t"]),
        "r2_train_a": float(m_a.rsquared),
        "r2_train_b": float(m_b.rsquared),
        "r2_test_a": r2(y_te, pred_a.values),
        "r2_test_b": r2(y_te, pred_b.values),
        "delta_r2_test": r2(y_te, pred_b.values) - r2(y_te, pred_a.values),
        "mae_test_a": mae_a,
        "mae_test_b": mae_b,
    }


def main():
    all_results = {}
    for exp in EXPERIMENTS:
        results = [run_config(c, exp) for c in CONFIGS]
        all_results[exp["name"]] = {"label": exp["label"], "results": results}
        print(f"\n=== H2 [{exp['label']}] ===\n")
        print(f"{'config':16s} {'tr_n':>5s} {'te_n':>5s} {'β_PE':>9s} {'p_PE':>7s} "
              f"{'R2_A_te':>8s} {'R2_B_te':>8s} {'ΔR2':>8s} {'MAE_B':>7s}")
        for r in results:
            print(f"{r['config']:16s} {r['train_n']:>5d} {r['test_n']:>5d} "
                  f"{r['beta_pe']:>+9.4f} {r['p_pe']:>7.3f} "
                  f"{r['r2_test_a']:>+8.3f} {r['r2_test_b']:>+8.3f} "
                  f"{r['delta_r2_test']:>+8.4f} {r['mae_test_b']:>7.4f}")

    with open(OUTPUTS / "e3_h2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
