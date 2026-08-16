"""E4 tarea 3.2: tablas y charts para la tesis.

Genera en outputs/charts/:
1. e4_mae_por_fold.png: MAE baseline vs con PE por fold (mejor config)
2. e4_delta_mae_ci.png: delta MAE con CI95 por config
3. e4_feature_importance.png: permutation importance (mejor config, ultimo fold)
4. e4_pred_vs_actual.png: predicho vs real OOS (mejor config)
Y la tabla comparativa final en consola / outputs/tabla_final.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
PREDS = OUT / "preds"
MODELS = OUT / "models"
CHARTS = OUT / "charts"

SEED = 42
TARGET = "trough_final"
BASE_FEATURES = ["drawdown_actual_t", "vol_realizada_t", "reversal_5d_t",
                  "market_dd_t", "rel_volume_t"]
CONFIG_LABELS = {"rf_w140": "RF, W=140", "rf_w180": "RF, W=180",
                  "gb_w140": "GB, W=140", "gb_w180": "GB, W=180",
                  "rf_wpe_w140": "RF, WPE W=140", "rf_wpe_w180": "RF, WPE W=180"}


def load_inference():
    with open(OUT / "inference.json") as f:
        return json.load(f)


def pick_best(inf):
    return min(inf, key=lambda c: inf[c]["point"]["mae_conpe"])


def tabla_final(inf):
    rows = []
    for c, d in inf.items():
        p = d["point"]
        rows.append({
            "config": CONFIG_LABELS.get(c, c),
            "MAE OLS base": p["mae_ols_baseline"],
            "MAE OLS PE": p["mae_ols_conpe"],
            "MAE base": p["mae_baseline"],
            "MAE con PE": p["mae_conpe"],
            "dMAE": p["delta_mae"],
            "dMAE CI lo": d["ci"]["ci_delta_mae"][0],
            "dMAE CI hi": d["ci"]["ci_delta_mae"][1],
            "R2 base": p["r2_baseline"],
            "R2 con PE": p["r2_conpe"],
            "folds mejora": f"{d['folds_pe_mejora']}/{d['n_folds']}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tabla_final.csv", index=False)
    print(df.round(4).to_string(index=False))
    return df


def chart_mae_por_fold(best):
    res = pd.read_csv(OUT / f"results_{best}.csv")
    x = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, res["mae_baseline"], 0.4, label="Baseline sin PE",
            color="grey", edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, res["mae_conpe"], 0.4, label="Con PE",
            color="steelblue", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(res["test_years"], fontsize=9)
    ax.set_xlabel("Bloque de test (walk-forward)")
    ax.set_ylabel("MAE OOS")
    ax.set_title(f"MAE por fold ({CONFIG_LABELS[best]})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHARTS / "e4_mae_por_fold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved e4_mae_por_fold.png")


def chart_delta_mae_ci(inf):
    configs = list(inf.keys())
    deltas = [inf[c]["point"]["delta_mae"] for c in configs]
    los = [inf[c]["ci"]["ci_delta_mae"][0] for c in configs]
    his = [inf[c]["ci"]["ci_delta_mae"][1] for c in configs]
    y = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(deltas, y,
                 xerr=[np.array(deltas) - np.array(los),
                       np.array(his) - np.array(deltas)],
                 fmt="o", color="steelblue", ecolor="grey",
                 elinewidth=2, capsize=4, markersize=8)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([CONFIG_LABELS[c] for c in configs])
    ax.set_xlabel("$\\Delta$MAE (con PE $-$ baseline), CI 95% bootstrap por episodio")
    ax.set_title("Aporte incremental de PE por configuración")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHARTS / "e4_delta_mae_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved e4_delta_mae_ci.png")


def chart_feature_importance(best):
    parts = best.split("_")
    if len(parts) == 3:              # rf_wpe_w140
        kind, var, w = parts
        model_stem = f"{kind}_{w}_{var}"
    else:                            # rf_w140
        kind, w = parts
        var = "pe"
        model_stem = f"{kind}_{w}"
    pe_col = f"{var}_t_{w}"
    feats = [pe_col] + BASE_FEATURES
    df = pd.read_parquet(DATA / "events_features.parquet")
    df = df.dropna(subset=feats + [TARGET])
    last_fold = 8
    test = df[df["year"].isin([2023, 2024])]
    m = joblib.load(MODELS / f"{model_stem}_f{last_fold}_conpe.joblib")
    pi = permutation_importance(m, test[feats].values, test[TARGET].values,
                                 n_repeats=20, random_state=SEED,
                                 scoring="neg_mean_absolute_error")
    order = np.argsort(pi.importances_mean)
    labels = [feats[i] for i in order]
    means = pi.importances_mean[order]
    stds = pi.importances_std[order]
    colors = ["steelblue" if l == pe_col else "grey" for l in labels]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(labels, means, xerr=stds, color=colors, edgecolor="black",
             linewidth=0.4)
    ax.set_xlabel("Permutation importance (aumento de MAE al permutar)")
    ax.set_title(f"Importancia de features ({CONFIG_LABELS[best]}, fold 2023-2024)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHARTS / "e4_feature_importance.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved e4_feature_importance.png")


def chart_pred_vs_actual(best):
    preds = pd.read_parquet(PREDS / f"preds_{best}.parquet")
    y = preds[TARGET].values
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(y, preds["pred_baseline"], s=8, alpha=0.25, c="grey",
                label="Baseline sin PE")
    ax.scatter(y, preds["pred_conpe"], s=8, alpha=0.25, c="steelblue",
                label="Con PE")
    lo = float(min(y.min(), preds["pred_baseline"].min(),
                    preds["pred_conpe"].min()))
    hi = float(max(y.max(), preds["pred_baseline"].max(),
                    preds["pred_conpe"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="predicción perfecta")
    ax.set_xlabel("Trough final real (OOS 2009-2024)")
    ax.set_ylabel("Trough final predicho")
    ax.set_title(f"Predicho vs. real ({CONFIG_LABELS[best]}, "
                  f"n = {len(preds)})")
    leg = ax.legend(fontsize=9, loc="best")
    for h in leg.legend_handles:
        if hasattr(h, "set_alpha"):
            h.set_alpha(1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHARTS / "e4_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved e4_pred_vs_actual.png")


def main():
    inf = load_inference()
    best = pick_best(inf)
    print(f"Mejor config por MAE con PE: {best}\n")
    tabla_final(inf)
    chart_mae_por_fold(best)
    chart_delta_mae_ci(inf)
    chart_feature_importance(best)
    chart_pred_vs_actual(best)


if __name__ == "__main__":
    main()
