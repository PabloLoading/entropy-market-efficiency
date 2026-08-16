"""H2 charts: dos figuras separadas.

1) e3_h2_pred_vs_actual: predicho vs. real en el test set (modelo con PE).
2) e3_h2_r2_bars: comparación de R^2 OOS con PE vs. sin PE, 4 configs.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from helpers import CHARTS, detect_events, load_price_ret_pe
from run_h2 import (
    CONFIGS, TRAIN_END, TEST_START, build_panel, fit_ols_cluster, run_config,
)

MAIN_CFG = {"label": "W=180, |dd| ≤ 25%", "W": 180, "cutoff": 0.25}


def prepare_panel(cfg):
    close, log_ret, pe = load_price_ret_pe(pe_window=cfg["W"], m=3)
    events = detect_events(close)
    panel = build_panel(events, close, log_ret, pe)
    if cfg["cutoff"] is not None:
        panel = panel[panel["event_max_dd"] <= cfg["cutoff"]].copy()
    panel = panel.dropna(subset=["pe_t", "drawdown_actual_t", "vol_realizada_t"])
    panel["peak_date"] = panel["event_id"].map(events["peak_date"])
    return panel


def plot_pred_vs_actual():
    panel = prepare_panel(MAIN_CFG)
    train = panel[panel["peak_date"] < TRAIN_END].copy()
    test = panel[panel["peak_date"] >= TEST_START].copy()

    m_a = fit_ols_cluster(train, ["drawdown_actual_t", "vol_realizada_t"])
    m_b = fit_ols_cluster(train, ["pe_t", "drawdown_actual_t", "vol_realizada_t"])

    early = (
        test[test["drawdown_actual_t"] <= -0.05]
        .sort_values("day")
        .groupby("event_id", as_index=False)
        .first()
    )
    y = early["trough_final"].values
    pred_a = m_a.predict(sm.add_constant(early[["drawdown_actual_t", "vol_realizada_t"]]))
    pred_b = m_b.predict(sm.add_constant(early[["pe_t", "drawdown_actual_t", "vol_realizada_t"]]))

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_a = 1 - np.sum((y - pred_a) ** 2) / ss_tot
    r2_b = 1 - np.sum((y - pred_b) ** 2) / ss_tot
    mae_a = float(np.mean(np.abs(y - pred_a)))
    mae_b = float(np.mean(np.abs(y - pred_b)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(y, pred_a, s=80, alpha=0.65, c="grey",
                edgecolor="black", linewidth=0.5,
                label=f"Sin PE ($R^2 = {r2_a:+.2f}$, MAE = {mae_a:.3f})")
    ax.scatter(y, pred_b, s=80, alpha=0.75, c="steelblue",
                edgecolor="black", linewidth=0.5,
                label=f"Con PE ($R^2 = {r2_b:+.2f}$, MAE = {mae_b:.3f})")
    lo = float(min(y.min(), pred_a.min(), pred_b.min()))
    hi = float(max(y.max(), pred_a.max(), pred_b.max()))
    margin = 0.02
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
             "k--", lw=1.2, label="predicción perfecta")
    ax.set_xlabel("Trough final real (test set)")
    ax.set_ylabel("Trough final predicho al inicio del evento")
    ax.set_title(f"H2: predicción temprana por evento (1 punto por evento)\n"
                  f"{MAIN_CFG['label']}, n = {len(early)} eventos, "
                  f"predicción al primer día con $|$dd$| \\geq 5\\%$")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHARTS / "e3_h2_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved e3_h2_pred_vs_actual.png (early R^2 A={r2_a:.3f}, B={r2_b:.3f})")


def plot_r2_bars():
    results = [run_config(c) for c in CONFIGS]
    labels = [r["config"].replace("_", "\n") for r in results]
    r2_a = [r["split"]["early_r2_baseline"] for r in results]
    r2_b = [r["split"]["early_r2_with_pe"] for r in results]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, r2_a, 0.4, label="Modelo sin PE",
            color="grey", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, r2_b, 0.4, label="Modelo con PE",
            color="steelblue", alpha=0.9, edgecolor="black", linewidth=0.4)
    for xi, (a, b) in enumerate(zip(r2_a, r2_b)):
        y_a = a + 0.02 if a >= 0 else a - 0.05
        y_b = b + 0.02 if b >= 0 else b - 0.05
        ax.text(xi - 0.2, y_a, f"{a:+.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, y_b, f"{b:+.2f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$R^2$ OOS temprano (1 punto por evento)")
    ax.set_title("H2: comparación de $R^2$ OOS temprano con vs. sin PE")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(min(r2_a), min(r2_b)) * 1.3, max(max(r2_a), max(r2_b)) * 1.3)
    plt.tight_layout()
    fig.savefig(CHARTS / "e3_h2_r2_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved e3_h2_r2_bars.png")


if __name__ == "__main__":
    plot_pred_vs_actual()
    plot_r2_bars()
