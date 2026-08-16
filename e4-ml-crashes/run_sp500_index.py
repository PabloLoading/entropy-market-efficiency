"""E4 sub-experimento: eventos del indice S&P 500, 1 asset.

Re-testea la senial de PE a nivel de mercado agregado (precursor 5.3/B.3.x)
con el rigor de E4: baseline rico, split OOS con embargo, CIs bootstrap.

Adaptaciones por 1 asset :
- ^GSPC 1928-2025, eventos dd >= 5% y duracion >= 15d; pre-1950 sin volumen
  de Yahoo se excluyen.
- Features E4 sin market_dd_t (el asset ES el mercado): baseline =
  drawdown_actual_t, vol_realizada_t, reversal_5d_t, rel_volume_t.
- Split unico: train peak < 2000, embargo 1 anio, test peak >= 2001.
- RF fijo (300, depth 6, leaf 5, seed 42) sin tuning; OLS referencia.
- Bootstrap simple sobre eventos de test (no se solapan; sin clusters).
- 4 runs W&B tag index-sp500: rf x {pe, wpe} x {W140, W180}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from entropy import perm_entropy  # noqa: E402

from eda_events import detect_events  # noqa: E402  (mismo detector que E4)

DATA = HERE / "data"
OUT = HERE / "outputs"
load_dotenv(HERE / ".env")  # solo necesario para W&B (opcional)

SEED = 42
PE_M = 3
PE_WINDOWS = [140, 180]
RELVOL_SHORT, RELVOL_LONG = 5, 60
MIN_RET = 5
TRAIN_END = pd.Timestamp("2000-01-01")
TEST_START = pd.Timestamp("2001-01-01")   # embargo 1 anio
N_BOOT = 2000
TARGET = "trough_final"
BASE_FEATURES = ["drawdown_actual_t", "vol_realizada_t", "reversal_5d_t",
                  "rel_volume_t"]


def load_gspc():
    f = DATA / "gspc_full.parquet"
    if not f.exists():
        import yfinance as yf
        df = yf.download("^GSPC", start="1928-01-01", end="2025-12-31",
                         interval="1d", auto_adjust=False, progress=False,
                         multi_level_index=False)
        df.to_parquet(f)
    return pd.read_parquet(f)


def build_events():
    px = load_gspc()
    close = px["Adj Close"].dropna()
    volume = px["Volume"].reindex(close.index)
    log_ret = np.log(close / close.shift(1)).dropna()

    ev = detect_events(close)
    ev = ev[ev["duration_days"] >= 15].copy()

    rows = []
    for _, e in ev.iterrows():
        entry, peak = e["entry_date"], e["peak_date"]
        ret_hist = log_ret.loc[:entry]
        row = {"peak_date": peak, "entry_date": entry,
               "trough_final": abs(e["drawdown"])}
        for w in PE_WINDOWS:
            win = ret_hist.iloc[-w:]
            if len(win) == w:
                row[f"pe_t_w{w}"] = perm_entropy(win.values, m=PE_M, tau=1)
                row[f"wpe_t_w{w}"] = perm_entropy(win.values, m=PE_M, tau=1,
                                                   weighted=True)
            else:
                row[f"pe_t_w{w}"] = row[f"wpe_t_w{w}"] = np.nan
        row["drawdown_actual_t"] = float(close.loc[entry] / close.loc[peak] - 1)
        rpe = log_ret.loc[peak:entry]
        if len(rpe) < MIN_RET:
            rpe = ret_hist.iloc[-MIN_RET:]
        row["vol_realizada_t"] = float(rpe.std()) if len(rpe) >= MIN_RET else np.nan
        row["reversal_5d_t"] = (float(ret_hist.iloc[-5:].sum())
                                 if len(ret_hist) >= 5 else np.nan)
        vol_hist = volume.loc[:entry].dropna()
        vol_hist = vol_hist[vol_hist > 0]
        if len(vol_hist) >= RELVOL_LONG:
            row["rel_volume_t"] = float(vol_hist.iloc[-RELVOL_SHORT:].mean()
                                         / vol_hist.iloc[-RELVOL_LONG:].mean())
        else:
            row["rel_volume_t"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def bootstrap_ci(y, pred_a, pred_b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas_mae, deltas_r2 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas_mae.append(mae(y[idx], pred_b[idx]) - mae(y[idx], pred_a[idx]))
        deltas_r2.append(r2(y[idx], pred_b[idx]) - r2(y[idx], pred_a[idx]))
    return ([float(np.percentile(deltas_mae, q)) for q in (2.5, 97.5)],
            [float(np.percentile(deltas_r2, q)) for q in (2.5, 97.5)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb

    df = build_events()
    print(f"Eventos del indice (>=15d): {len(df)}")

    results = {}
    for variant in ["pe", "wpe"]:
        for w in PE_WINDOWS:
            col = f"{variant}_t_w{w}"
            feats_ent = [col] + BASE_FEATURES
            d = df.dropna(subset=feats_ent + [TARGET]).copy()
            train = d[d["peak_date"] < TRAIN_END]
            test = d[d["peak_date"] >= TEST_START]
            config = f"sp500_rf_{variant}_w{w}" if variant == "wpe" else f"sp500_rf_w{w}"
            name = f"sp500_rf_{'wpe_' if variant == 'wpe' else ''}w{w}"

            y_tr, y_te = train[TARGET].values, test[TARGET].values
            out = {"config": name, "n_train": len(train), "n_test": len(test)}
            preds = {}
            for label, feats in [("baseline", BASE_FEATURES),
                                  ("conpe", feats_ent)]:
                m = RandomForestRegressor(n_estimators=300, max_depth=6,
                                           min_samples_leaf=5,
                                           random_state=SEED, n_jobs=-1)
                m.fit(train[feats].values, y_tr)
                p = m.predict(test[feats].values)
                preds[label] = p
                out[f"mae_{label}"] = mae(y_te, p)
                out[f"r2_{label}"] = r2(y_te, p)
                ols = LinearRegression().fit(train[feats].values, y_tr)
                p_ols = ols.predict(test[feats].values)
                out[f"mae_ols_{label}"] = mae(y_te, p_ols)
                out[f"r2_ols_{label}"] = r2(y_te, p_ols)

            out["delta_mae"] = out["mae_conpe"] - out["mae_baseline"]
            out["delta_r2"] = out["r2_conpe"] - out["r2_baseline"]
            ci_mae, ci_r2 = bootstrap_ci(y_te, preds["baseline"], preds["conpe"])
            out["ci_delta_mae"] = ci_mae
            out["ci_delta_r2"] = ci_r2
            results[name] = out

            print(f"\n=== {name} (train={out['n_train']}, test={out['n_test']}) ===")
            print(f"  MAE base {out['mae_baseline']:.4f}  con {variant.upper()} "
                  f"{out['mae_conpe']:.4f}  dMAE {out['delta_mae']:+.4f} "
                  f"CI[{ci_mae[0]:+.4f},{ci_mae[1]:+.4f}]")
            print(f"  R2  base {out['r2_baseline']:+.4f}  con {variant.upper()} "
                  f"{out['r2_conpe']:+.4f}  dR2 {out['delta_r2']:+.4f} "
                  f"CI[{ci_r2[0]:+.4f},{ci_r2[1]:+.4f}]")

            if wandb is not None:
                run = wandb.init(project="e4-pe-ml-crashes", name=name,
                                  group=name, tags=["index-sp500"],
                                  config={"variant": variant, "pe_window": w,
                                          "features_baseline": BASE_FEATURES,
                                          "rf_params": "n300_d6_l5",
                                          "split": "train<2000, embargo 1y, test>=2001",
                                          "seed": SEED},
                                  reinit=True)
                for k, v in out.items():
                    if isinstance(v, (int, float)):
                        run.summary[k] = v
                run.summary["ci_delta_mae_lo"] = ci_mae[0]
                run.summary["ci_delta_mae_hi"] = ci_mae[1]
                run.summary["ci_delta_r2_lo"] = ci_r2[0]
                run.summary["ci_delta_r2_hi"] = ci_r2[1]
                run.finish()

    with open(OUT / "sp500_index_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado {OUT / 'sp500_index_results.json'}")


if __name__ == "__main__":
    main()
