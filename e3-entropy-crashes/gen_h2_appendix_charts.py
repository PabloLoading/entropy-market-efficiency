"""Charts para las tablas del apéndice B.3.2 y B.3.3.

Genera 4 figuras:
1. e3_h2_multi_bars: bars R^2 baseline vs con PE (14 activos, split 2015/2017)
2. e3_h2_gspc_bars: bars R^2 baseline vs con PE (solo ^GSPC, split 2000/2002)
3. e3_h2_multi_pred: predicted vs actual mejor config multi-asset
4. e3_h2_gspc_pred: predicted vs actual mejor config solo ^GSPC
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from helpers import CHARTS
from run_h2 import CONFIGS, EXPERIMENTS, build_multi_asset_early, run_config


BEST_CONFIG_MULTI = "W=180_leq25"  # p_PE=0.003 (más significativa)
BEST_CONFIG_GSPC = "W=140_leq25"   # ΔR²=+0.105 (mejor mejora)


def get_experiment(name):
    return next(e for e in EXPERIMENTS if e["name"] == name)


def get_config(label):
    return next(c for c in CONFIGS if c["label"] == label)


def plot_bars(results, title, out_path):
    labels = [r["config"].replace("_", "\n") for r in results]
    r2_a = [r["r2_test_a"] for r in results]
    r2_b = [r["r2_test_b"] for r in results]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, r2_a, 0.4, label="Modelo baseline",
            color="grey", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, r2_b, 0.4, label="Modelo con PE",
            color="steelblue", alpha=0.9, edgecolor="black", linewidth=0.4)
    for xi, (a, b) in enumerate(zip(r2_a, r2_b)):
        ax.text(xi - 0.2, a + 0.003, f"{a:+.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, b + 0.003, f"{b:+.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$R^2$ OOS test")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    y_max = max(max(r2_a), max(r2_b)) * 1.2
    y_min = min(min(r2_a), min(r2_b), 0) * 1.3
    ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_pred_vs_actual(experiment, config_label, out_path):
    exp = get_experiment(experiment)
    cfg = get_config(config_label)
    df = build_multi_asset_early(pe_window=cfg["W"], m=3, assets=exp["assets"])
    df = df.dropna(subset=["pe_t", "drawdown_actual_t", "vol_realizada_t"])
    if cfg["cutoff"] is not None:
        df = df[df["event_max_dd"] <= cfg["cutoff"]].copy()
    train = df[df["peak_date"] < exp["train_end"]].copy()
    test = df[df["peak_date"] >= exp["test_start"]].copy()

    cols_a = ["drawdown_actual_t", "vol_realizada_t"]
    cols_b = ["pe_t", "drawdown_actual_t", "vol_realizada_t"]
    y_tr = train["trough_final"]
    m_a = sm.OLS(y_tr, sm.add_constant(train[cols_a])).fit()
    m_b = sm.OLS(y_tr, sm.add_constant(train[cols_b])).fit()

    y_te = test["trough_final"].values
    pred_a = m_a.predict(sm.add_constant(test[cols_a])).values
    pred_b = m_b.predict(sm.add_constant(test[cols_b])).values

    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    r2_a = 1 - np.sum((y_te - pred_a) ** 2) / ss_tot
    r2_b = 1 - np.sum((y_te - pred_b) ** 2) / ss_tot
    mae_a = float(np.mean(np.abs(y_te - pred_a)))
    mae_b = float(np.mean(np.abs(y_te - pred_b)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(y_te, pred_a, s=80, alpha=0.65, c="grey",
                edgecolor="black", linewidth=0.5,
                label=f"Baseline ($R^2 = {r2_a:+.2f}$, MAE = {mae_a:.3f})")
    ax.scatter(y_te, pred_b, s=80, alpha=0.75, c="steelblue",
                edgecolor="black", linewidth=0.5,
                label=f"Con PE ($R^2 = {r2_b:+.2f}$, MAE = {mae_b:.3f})")
    lo = float(min(y_te.min(), pred_a.min(), pred_b.min()))
    hi = float(max(y_te.max(), pred_a.max(), pred_b.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="predicción perfecta")
    ax.set_xlabel("Trough final real (test set)")
    ax.set_ylabel("Trough final predicho")
    ax.set_title(f"H2 [{exp['label']}]: {config_label} (n = {len(test)} eventos)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def main():
    for exp in EXPERIMENTS:
        results = [run_config(c, exp) for c in CONFIGS]
        prefix = "multi" if exp["name"] == "multi_asset_new" else "gspc"
        title = f"H2 [{exp['label']}]: $R^2$ OOS baseline vs con PE"
        plot_bars(results, title, CHARTS / f"e3_h2_{prefix}_bars.png")

    plot_pred_vs_actual("multi_asset_new", BEST_CONFIG_MULTI,
                        CHARTS / "e3_h2_multi_pred.png")
    plot_pred_vs_actual("gspc_only_old", BEST_CONFIG_GSPC,
                        CHARTS / "e3_h2_gspc_pred.png")


if __name__ == "__main__":
    main()
