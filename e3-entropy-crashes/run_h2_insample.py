"""H2 in-sample simplificado: solo ^GSPC, sin split.

Regresión OLS: trough_final ~ pe_t + drawdown_actual_t + vol_realizada_t.
Comparación modelo A (sin PE) vs modelo B (con PE), in-sample sobre todo el rango.
Una observación por evento (primer día con drawdown <= -5%).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from helpers import CHARTS, OUTPUTS, detect_events, load_price_ret_pe
from run_h2 import CONFIGS, build_early_obs


CHART_CONFIG = "W=180_leq25"


def run_config(cfg):
    close, log_ret, pe = load_price_ret_pe(pe_window=cfg["W"], m=3)
    events = detect_events(close)
    df = build_early_obs(events, close, log_ret, pe, ticker="^GSPC")
    df = df.dropna(subset=["pe_t", "drawdown_actual_t", "vol_realizada_t"])
    if cfg["cutoff"] is not None:
        df = df[df["event_max_dd"] <= cfg["cutoff"]].copy()

    y = df["trough_final"]
    cols_a = ["drawdown_actual_t", "vol_realizada_t"]
    cols_b = ["pe_t"] + cols_a

    m_a = sm.OLS(y, sm.add_constant(df[cols_a])).fit()
    m_b = sm.OLS(y, sm.add_constant(df[cols_b])).fit()

    mae_a = float(np.mean(np.abs(y.values - m_a.fittedvalues.values)))
    mae_b = float(np.mean(np.abs(y.values - m_b.fittedvalues.values)))

    return {
        "config": cfg["label"],
        "n": int(len(df)),
        "beta_pe": float(m_b.params["pe_t"]),
        "p_pe": float(m_b.pvalues["pe_t"]),
        "r2_a": float(m_a.rsquared),
        "r2_b": float(m_b.rsquared),
        "delta_r2": float(m_b.rsquared - m_a.rsquared),
        "mae_a": mae_a,
        "mae_b": mae_b,
        "delta_mae": mae_b - mae_a,
    }


def plot_pred_vs_actual(cfg, out_path):
    close, log_ret, pe = load_price_ret_pe(pe_window=cfg["W"], m=3)
    events = detect_events(close)
    df = build_early_obs(events, close, log_ret, pe, ticker="^GSPC")
    df = df.dropna(subset=["pe_t", "drawdown_actual_t", "vol_realizada_t"])
    if cfg["cutoff"] is not None:
        df = df[df["event_max_dd"] <= cfg["cutoff"]].copy()

    y = df["trough_final"].values
    cols_a = ["drawdown_actual_t", "vol_realizada_t"]
    cols_b = ["pe_t"] + cols_a
    m_a = sm.OLS(y, sm.add_constant(df[cols_a])).fit()
    m_b = sm.OLS(y, sm.add_constant(df[cols_b])).fit()
    pred_a = m_a.fittedvalues
    pred_b = m_b.fittedvalues

    r2_a = float(m_a.rsquared)
    r2_b = float(m_b.rsquared)
    mae_a = float(np.mean(np.abs(y - pred_a)))
    mae_b = float(np.mean(np.abs(y - pred_b)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(y, pred_a, s=70, alpha=0.6, c="grey",
                edgecolor="black", linewidth=0.4,
                label=f"Baseline sin PE ($R^2={r2_a:+.2f}$, MAE={mae_a:.3f})")
    ax.scatter(y, pred_b, s=70, alpha=0.75, c="steelblue",
                edgecolor="black", linewidth=0.4,
                label=f"Con PE ($R^2={r2_b:+.2f}$, MAE={mae_b:.3f})")
    lo = float(min(y.min(), pred_a.min(), pred_b.min()))
    hi = float(max(y.max(), pred_a.max(), pred_b.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="predicción perfecta")
    ax.set_xlabel("Trough final real")
    ax.set_ylabel("Trough final predicho (OLS)")
    ax.set_title(f"H2 in-sample (S&P 500, $W=180$, $\\leq 25\\%$, n = {len(df)})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def main():
    results = [run_config(c) for c in CONFIGS]
    print("\n=== H2 in-sample [^GSPC, sin split, todo rango] ===\n")
    print(f"{'config':16s} {'n':>4s} {'β_PE':>9s} {'p_PE':>7s} "
          f"{'R²_A':>7s} {'R²_B':>7s} {'ΔR²':>8s} {'MAE_A':>7s} {'MAE_B':>7s} {'ΔMAE':>8s}")
    for r in results:
        print(f"{r['config']:16s} {r['n']:>4d} "
              f"{r['beta_pe']:>+9.4f} {r['p_pe']:>7.3f} "
              f"{r['r2_a']:>+7.3f} {r['r2_b']:>+7.3f} "
              f"{r['delta_r2']:>+8.4f} "
              f"{r['mae_a']:>7.4f} {r['mae_b']:>7.4f} {r['delta_mae']:>+8.4f}")

    with open(OUTPUTS / "e3_h2_insample_results.json", "w") as f:
        json.dump(results, f, indent=2)

    cfg = next(c for c in CONFIGS if c["label"] == CHART_CONFIG)
    plot_pred_vs_actual(cfg, CHARTS / "e3_h2_insample_pred.png")


if __name__ == "__main__":
    main()
