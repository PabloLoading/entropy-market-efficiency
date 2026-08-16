"""H2 main chart: 3 panels visualizing PE marginal contribution.

A) Partial residual plot: aporte marginal de PE (in-sample, config principal).
B) Predicted vs actual OOS: aplicabilidad en test set con Modelo A vs Modelo B.
C) Bar chart R^2 OOS Modelo A vs Modelo B sobre las 4 configs.
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
    peak_dates = events["peak_date"]
    panel["peak_date"] = panel["event_id"].map(peak_dates)
    return panel


def draw_partial_residual(ax, panel):
    X = sm.add_constant(panel[["drawdown_actual_t", "vol_realizada_t"]])
    y = panel["trough_final"]
    pe = panel["pe_t"]
    resid_y = y - sm.OLS(y, X).fit().predict(X)
    resid_x = pe - sm.OLS(pe, X).fit().predict(X)
    ax.scatter(resid_x, resid_y, alpha=0.15, s=8, c="steelblue")
    slope, intercept = np.polyfit(resid_x, resid_y, 1)
    xr = np.linspace(resid_x.min(), resid_x.max(), 100)
    ax.plot(xr, slope * xr + intercept, "k--", lw=1.4,
             label=f"pendiente = {slope:+.3f}")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Residuos de PE_t (tras dd, vol)")
    ax.set_ylabel("Residuos de trough final (tras dd, vol)")
    ax.set_title("A. Aporte marginal de PE\n(partial residual)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def draw_predicted_vs_actual(ax, train, test):
    m_a = fit_ols_cluster(train, ["drawdown_actual_t", "vol_realizada_t"])
    m_b = fit_ols_cluster(train, ["pe_t", "drawdown_actual_t", "vol_realizada_t"])
    y = test["trough_final"]
    X_a = sm.add_constant(test[["drawdown_actual_t", "vol_realizada_t"]])
    X_b = sm.add_constant(test[["pe_t", "drawdown_actual_t", "vol_realizada_t"]])
    pred_a = m_a.predict(X_a)
    pred_b = m_b.predict(X_b)
    ax.scatter(y, pred_a, s=14, alpha=0.35, c="grey", label="Sin PE")
    ax.scatter(y, pred_b, s=14, alpha=0.55, c="steelblue", label="Con PE")
    lo = float(min(y.min(), pred_a.min(), pred_b.min()))
    hi = float(max(y.max(), pred_a.max(), pred_b.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="predicción perfecta")
    ax.set_xlabel("trough real")
    ax.set_ylabel("trough predicho")
    ax.set_title("B. Predicho vs. real\n(OOS test set)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)


def draw_r2_bars(ax, results):
    labels = [r["config"].replace("_", "\n") for r in results]
    r2_a = [r["split"]["oos_r2_baseline"] for r in results]
    r2_b = [r["split"]["oos_r2_with_pe"] for r in results]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, r2_a, 0.4, label="Sin PE", color="grey", alpha=0.85)
    ax.bar(x + 0.2, r2_b, 0.4, label="Con PE", color="steelblue", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$R^2$ OOS")
    ax.set_title("C. $R^2$ OOS por config\n(con PE vs. sin PE)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def main():
    results = [run_config(c) for c in CONFIGS]
    panel = prepare_panel(MAIN_CFG)
    train = panel[panel["peak_date"] < TRAIN_END].copy()
    test = panel[panel["peak_date"] >= TEST_START].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    draw_partial_residual(axes[0], panel)
    draw_predicted_vs_actual(axes[1], train, test)
    draw_r2_bars(axes[2], results)
    fig.suptitle(f"H2 ({MAIN_CFG['label']}): aporte marginal de PE sobre trough final",
                  fontsize=12)
    plt.tight_layout()
    fig.savefig(CHARTS / "e3_h2_main.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved e3_h2_main.png")


if __name__ == "__main__":
    main()
