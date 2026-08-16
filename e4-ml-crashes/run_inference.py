"""E4 tarea 3.1: inferencia con bootstrap por cluster de episodio.

CIs 95% de MAE (baseline, con PE), delta MAE, R2 y delta R2 OOS, remuestreando
episodios enteros (eventos del mismo episodio sistemico entran o salen juntos;
los idiosincraticos son clusters unitarios). Reporta ademas metricas por fold
y por episodio sistemico.

Entrada: outputs/preds/preds_<config>.parquet (de run_models.py).
Salida: outputs/inference_<config>.json + tabla resumen en consola.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PREDS = OUT / "preds"

SEED = 42
N_BOOT = 2000
TARGET = "trough_final"

def discover_configs():
    """Todas las configs con preds guardadas (excluye sensibilidad no-weights)."""
    names = [f.stem.replace("preds_", "") for f in sorted(PREDS.glob("preds_*.parquet"))]
    return [n for n in names if "noweights" not in n]


CONFIGS = discover_configs()


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def cluster_bootstrap(df: pd.DataFrame, n_boot=N_BOOT, seed=SEED):
    """Bootstrap por episodio: remuestrea episode_ids con reemplazo.

    Implementado con indices numpy (remuestrear ~14k clusters via concat de
    DataFrames por replica es prohibitivo).
    """
    rng = np.random.default_rng(seed)
    y_all = df[TARGET].values
    pb_all = df["pred_baseline"].values
    pp_all = df["pred_conpe"].values
    codes, _ = pd.factorize(df["episode_id"])
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(sorted_codes.max() + 1))
    ends = np.append(starts[1:], len(sorted_codes))
    idx_by_ep = [order[s:e] for s, e in zip(starts, ends)]
    n_eps = len(idx_by_ep)

    stats = []
    for _ in range(n_boot):
        eps = rng.integers(0, n_eps, size=n_eps)
        idx = np.concatenate([idx_by_ep[e] for e in eps])
        y, pb, pp = y_all[idx], pb_all[idx], pp_all[idx]
        stats.append({
            "mae_baseline": mae(y, pb),
            "mae_conpe": mae(y, pp),
            "r2_baseline": r2(y, pb),
            "r2_conpe": r2(y, pp),
        })
    bs = pd.DataFrame(stats)
    bs["delta_mae"] = bs["mae_conpe"] - bs["mae_baseline"]
    bs["delta_r2"] = bs["r2_conpe"] - bs["r2_baseline"]
    return bs


def ci(series, lo=2.5, hi=97.5):
    return [float(np.percentile(series, lo)), float(np.percentile(series, hi))]


def main():
    summary = {}
    for config in CONFIGS:
        f = PREDS / f"preds_{config}.parquet"
        if not f.exists():
            print(f"(skip {config}: sin preds)")
            continue
        df = pd.read_parquet(f)
        y = df[TARGET].values

        point = {
            "n_events": int(len(df)),
            "n_episodes": int(df["episode_id"].nunique()),
            "mae_baseline": mae(y, df["pred_baseline"].values),
            "mae_conpe": mae(y, df["pred_conpe"].values),
            "r2_baseline": r2(y, df["pred_baseline"].values),
            "r2_conpe": r2(y, df["pred_conpe"].values),
            "mae_ols_baseline": mae(y, df["pred_ols_baseline"].values),
            "mae_ols_conpe": mae(y, df["pred_ols_conpe"].values),
            "r2_ols_baseline": r2(y, df["pred_ols_baseline"].values),
            "r2_ols_conpe": r2(y, df["pred_ols_conpe"].values),
        }
        point["delta_mae"] = point["mae_conpe"] - point["mae_baseline"]
        point["delta_r2"] = point["r2_conpe"] - point["r2_baseline"]

        bs = cluster_bootstrap(df)
        cis = {f"ci_{c}": ci(bs[c]) for c in
               ["mae_baseline", "mae_conpe", "delta_mae",
                "r2_baseline", "r2_conpe", "delta_r2"]}

        by_fold = df.groupby("fold").apply(
            lambda g: pd.Series({
                "mae_baseline": mae(g[TARGET].values, g["pred_baseline"].values),
                "mae_conpe": mae(g[TARGET].values, g["pred_conpe"].values),
            }), include_groups=False)
        by_fold["delta_mae"] = by_fold["mae_conpe"] - by_fold["mae_baseline"]

        sys_df = df[df["systemic"]]
        by_episode = sys_df.groupby("episode_id").apply(
            lambda g: pd.Series({
                "n": len(g),
                "mae_baseline": mae(g[TARGET].values, g["pred_baseline"].values),
                "mae_conpe": mae(g[TARGET].values, g["pred_conpe"].values),
            }), include_groups=False) if len(sys_df) else pd.DataFrame()

        summary[config] = {
            "point": point, "ci": cis,
            "folds_pe_mejora": int((by_fold["delta_mae"] < 0).sum()),
            "n_folds": int(len(by_fold)),
            "by_fold": by_fold.round(5).to_dict(orient="index"),
            "by_episode": (by_episode.round(5).to_dict(orient="index")
                            if len(by_episode) else {}),
        }

        d = point["delta_mae"]
        lo_d, hi_d = cis["ci_delta_mae"]
        print(f"\n=== {config} (n={point['n_events']}, "
              f"episodios={point['n_episodes']}) ===")
        print(f"  MAE baseline {point['mae_baseline']:.4f}  "
              f"con PE {point['mae_conpe']:.4f}  "
              f"OLS ref {point['mae_ols_baseline']:.4f}/{point['mae_ols_conpe']:.4f}")
        print(f"  R2  baseline {point['r2_baseline']:.4f}  "
              f"con PE {point['r2_conpe']:.4f}")
        print(f"  dMAE {d:+.5f}  CI95 [{lo_d:+.5f}, {hi_d:+.5f}]  "
              f"{'PE mejora' if hi_d < 0 else ('PE empeora' if lo_d > 0 else 'CI cruza cero')}")
        print(f"  folds con mejora PE: {summary[config]['folds_pe_mejora']}"
              f"/{summary[config]['n_folds']}")

    with open(OUT / "inference.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGuardado {OUT / 'inference.json'}")


if __name__ == "__main__":
    main()
