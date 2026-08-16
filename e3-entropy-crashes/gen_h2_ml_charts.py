"""Charts para el apéndice B.3.4 (H2 enhanced con Random Forest, solo GSPC).

Genera 2 figuras:
1. e3_h2_gspc_ml_fi: feature importance (solo GSPC, W=180_leq25)
2. e3_h2_gspc_ml_pred: predicted vs actual (solo GSPC, W=180_all, mejor R^2)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from helpers import CHARTS
from run_h2 import CONFIGS, EXPERIMENTS
from run_h2_ml import build_multi_asset_enhanced, fit_rf


FI_CONFIG = "W=180_leq25"
PRED_CONFIG = "W=180_all"


def get_experiment(name):
    return next(e for e in EXPERIMENTS if e["name"] == name)


def get_config(label):
    return next(c for c in CONFIGS if c["label"] == label)


def prep_split(experiment, config_label):
    exp = get_experiment(experiment)
    cfg = get_config(config_label)
    df = build_multi_asset_enhanced(pe_window=cfg["W"], m=3, assets=exp["assets"])
    df = df.dropna(subset=["pe_t", "drawdown_actual_t", "vol_realizada_t",
                             "vol_pre_peak", "reversal_5d_t"])
    if cfg["cutoff"] is not None:
        df = df[df["event_max_dd"] <= cfg["cutoff"]].copy()
    train = df[df["peak_date"] < exp["train_end"]].copy()
    test = df[df["peak_date"] >= exp["test_start"]].copy()
    return train, test


def plot_feature_importance(experiment, config_label, out_path, title_suffix):
    train, _ = prep_split(experiment, config_label)
    cols = ["pe_t", "drawdown_actual_t", "vol_realizada_t",
             "vol_pre_peak", "reversal_5d_t"]
    m = fit_rf(train[cols].values, train["trough_final"].values)
    fi = dict(zip(cols, m.feature_importances_))
    ordered = sorted(fi.items(), key=lambda x: x[1])
    labels = [k for k, _ in ordered]
    values = [v for _, v in ordered]
    colors = ["steelblue" if k == "pe_t" else "grey" for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, values):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                 f"{v * 100:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Feature importance (Random Forest)")
    ax.set_title(f"H2 enhanced — importancia de features ({title_suffix})")
    ax.set_xlim(0, max(values) * 1.25)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_pred_vs_actual_ml(experiment, config_label, out_path, title_suffix):
    train, test = prep_split(experiment, config_label)
    cols_a = ["drawdown_actual_t", "vol_realizada_t", "vol_pre_peak", "reversal_5d_t"]
    cols_b = ["pe_t"] + cols_a
    y_tr = train["trough_final"].values
    m_a = fit_rf(train[cols_a].values, y_tr)
    m_b = fit_rf(train[cols_b].values, y_tr)

    y_te = test["trough_final"].values
    pred_a = m_a.predict(test[cols_a].values)
    pred_b = m_b.predict(test[cols_b].values)

    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    r2_a = 1 - np.sum((y_te - pred_a) ** 2) / ss_tot
    r2_b = 1 - np.sum((y_te - pred_b) ** 2) / ss_tot
    mae_a = float(np.mean(np.abs(y_te - pred_a)))
    mae_b = float(np.mean(np.abs(y_te - pred_b)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(y_te, pred_a, s=80, alpha=0.6, c="grey",
                edgecolor="black", linewidth=0.4,
                label=f"Baseline sin PE ($R^2={r2_a:+.2f}$, MAE={mae_a:.3f})")
    ax.scatter(y_te, pred_b, s=80, alpha=0.75, c="steelblue",
                edgecolor="black", linewidth=0.4,
                label=f"Con PE ($R^2={r2_b:+.2f}$, MAE={mae_b:.3f})")
    lo = float(min(y_te.min(), pred_a.min(), pred_b.min()))
    hi = float(max(y_te.max(), pred_a.max(), pred_b.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="predicción perfecta")
    ax.set_xlabel("Trough final real (test set)")
    ax.set_ylabel("Trough final predicho (Random Forest)")
    ax.set_title(f"H2 enhanced — predicho vs. real ({title_suffix}, n = {len(test)})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def main():
    plot_feature_importance("gspc_only_old", FI_CONFIG,
                              CHARTS / "e3_h2_gspc_ml_fi.png",
                              "solo GSPC, $W=180$, $\\leq 25\\%$")
    plot_pred_vs_actual_ml("gspc_only_old", PRED_CONFIG,
                             CHARTS / "e3_h2_gspc_ml_pred.png",
                             "solo GSPC, $W=180$, sin corte")


if __name__ == "__main__":
    main()
