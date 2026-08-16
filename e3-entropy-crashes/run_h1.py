"""H1: relación entre drawdown magnitude y caída de PE."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from helpers import (
    CHARTS, CRASH_CUTOFF, MIN_EVENT_DAYS_MAIN, MIN_EVENT_DAYS_SENS,
    OUTPUTS, PE_MAIN, PE_SENS,
    classify, compute_event_metrics, detect_events, load_price_ret_pe,
)


def fit_ols(df, cols, cluster=None):
    y = df["delta_pe"]
    X = sm.add_constant(df[cols])
    if cluster is not None:
        return sm.OLS(y, X).fit(cov_type="cluster",
                                 cov_kwds={"groups": df[cluster]})
    return sm.OLS(y, X).fit(cov_type="HC3")


def summarize(model, target="drawdown"):
    return {
        "coef_target": float(model.params[target]),
        "p_target": float(model.pvalues[target]),
        "r2": float(model.rsquared),
        "n": int(model.nobs),
        "coef_vol": float(model.params.get("vol_pre_peak", np.nan)),
        "p_vol": float(model.pvalues.get("vol_pre_peak", np.nan)),
        "coef_pe_pre": float(model.params.get("pe_pre_peak", np.nan)),
        "p_pe_pre": float(model.pvalues.get("pe_pre_peak", np.nan)),
    }


def plot_scatter(df, model, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    colors = {"pullback": "steelblue", "crash": "crimson"}
    for cat in ["pullback", "crash"]:
        sub = df[df["type"] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(sub["drawdown"].abs() * 100, sub["delta_pe"],
                    c=colors[cat], label=f"{cat} (n={len(sub)})", s=42,
                    alpha=0.75, edgecolor="black", linewidth=0.4)
        if len(sub) >= 3:
            m_sub = fit_ols(sub, ["drawdown"])
            xr = np.linspace(sub["drawdown"].min(), sub["drawdown"].max(), 50)
            yr = m_sub.params["const"] + m_sub.params["drawdown"] * xr
            ax.plot(-xr * 100, yr, "--", color=colors[cat], lw=1.3, alpha=0.9,
                     label=f"OLS {cat}: $\\beta={m_sub.params['drawdown']:+.3f}$")

    xr = np.linspace(df["drawdown"].min(), df["drawdown"].max(), 100)
    yr = model.params["const"] + model.params["drawdown"] * xr
    ax.plot(-xr * 100, yr, "k--", lw=1.4,
             label=f"OLS todos: $\\beta={model.params['drawdown']:+.3f}$, "
                   f"$R^2={model.rsquared:.2f}$")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Caída de mercado ($|$drawdown$|$, \\%)")
    ax.set_ylabel("Caída de PE (unidades absolutas)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def interaction_test(df):
    df = df.copy()
    df["is_crash"] = (df["type"] == "crash").astype(int)
    df["dd_x_crash"] = df["drawdown"] * df["is_crash"]
    y = df["delta_pe"]
    X = sm.add_constant(df[["drawdown", "is_crash", "dd_x_crash"]])
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["decade"]})
    return {
        "coef_interaction": float(m.params["dd_x_crash"]),
        "p_interaction": float(m.pvalues["dd_x_crash"]),
        "coef_drawdown_pullback": float(m.params["drawdown"]),
        "coef_drawdown_crash": float(m.params["drawdown"] + m.params["dd_x_crash"]),
        "r2": float(m.rsquared),
        "n": int(m.nobs),
    }


def run_config(pe_window, m, min_event_days):
    close, log_ret, pe = load_price_ret_pe(pe_window=pe_window, m=m)
    events = detect_events(close)
    df = compute_event_metrics(events, log_ret, pe, pe_window=pe_window,
                                min_event_days=min_event_days)
    df["type"] = classify(df["drawdown"], cutoff=CRASH_CUTOFF)

    models = {
        "simple_hc3": fit_ols(df, ["drawdown"]),
        "simple_cluster": fit_ols(df, ["drawdown"], cluster="decade"),
        "vol_control": fit_ols(df, ["drawdown", "vol_pre_peak"], cluster="decade"),
        "full_control": fit_ols(df, ["drawdown", "vol_pre_peak", "pe_pre_peak"],
                                 cluster="decade"),
    }
    return df, models


def main():
    df_main, models_main = run_config(PE_MAIN["window"], PE_MAIN["m"],
                                       MIN_EVENT_DAYS_MAIN)
    plot_scatter(df_main, models_main["simple_hc3"],
                  CHARTS / "e3_h1_scatter_main.png",
                  f"H1: drawdown vs. caída de PE (main, mín. {MIN_EVENT_DAYS_MAIN}d)")
    df_main.to_csv(OUTPUTS / "e3_h1_events_main.csv", index=False)

    results = {
        "main": {"config": {"m": PE_MAIN["m"], "window": PE_MAIN["window"],
                              "crash_cutoff": CRASH_CUTOFF,
                              "min_event_days": MIN_EVENT_DAYS_MAIN,
                              "n": len(df_main)},
                  "models": {k: summarize(v) for k, v in models_main.items()},
                  "interaction": interaction_test(df_main)},
        "sensitivity": {},
    }

    for cfg in PE_SENS:
        df_s, m_sens = run_config(cfg["window"], cfg["m"], MIN_EVENT_DAYS_MAIN)
        label = f"m{cfg['m']}_w{cfg['window']}"
        results["sensitivity"][label] = {
            "models": {k: summarize(v) for k, v in m_sens.items()},
            "interaction": interaction_test(df_s),
        }
        plot_scatter(df_s, m_sens["simple_hc3"],
                      CHARTS / f"e3_h1_scatter_{label}.png",
                      f"H1: {label} (mín. {MIN_EVENT_DAYS_MAIN}d)")

    for d in MIN_EVENT_DAYS_SENS:
        df_s, m_sens = run_config(PE_MAIN["window"], PE_MAIN["m"], d)
        label = f"min{d}d"
        results["sensitivity"][label] = {
            "models": {k: summarize(v) for k, v in m_sens.items()},
            "interaction": interaction_test(df_s),
        }
        plot_scatter(df_s, m_sens["simple_hc3"],
                      CHARTS / f"e3_h1_scatter_{label}.png",
                      f"H1: filtro duración mín. {d} días")

    for label_prefix, (pe_w, pe_m) in [
        ("dd_leq_30", (PE_MAIN["window"], PE_MAIN["m"])),
        ("dd_leq_30_w180", (180, 3)),
    ]:
        df_full, _ = run_config(pe_w, pe_m, MIN_EVENT_DAYS_MAIN)
        df_sub = df_full[df_full["drawdown"].abs() <= 0.30].copy()
        m_sub = {
            "simple_hc3": fit_ols(df_sub, ["drawdown"]),
            "vol_control": fit_ols(df_sub, ["drawdown", "vol_pre_peak"], cluster="decade"),
            "full_control": fit_ols(df_sub, ["drawdown", "vol_pre_peak", "pe_pre_peak"],
                                      cluster="decade"),
        }
        results["sensitivity"][label_prefix] = {
            "models": {k: summarize(v) for k, v in m_sub.items()},
            "interaction": interaction_test(df_sub),
        }
        suffix = " (w=180)" if "w180" in label_prefix else ""
        plot_scatter(df_sub, m_sub["simple_hc3"],
                      CHARTS / f"e3_h1_scatter_{label_prefix}.png",
                      f"H1: subset $|$drawdown$| \\leq 30\\%${suffix}")

    with open(OUTPUTS / "e3_h1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Main config (mín. {MIN_EVENT_DAYS_MAIN}d): n_events={len(df_main)}")
    for k, v in results["main"]["models"].items():
        print(f"  {k:16s} coef={v['coef_target']:+.4f} p={v['p_target']:.4f} "
              f"R2={v['r2']:.3f} n={v['n']}")
    it = results["main"]["interaction"]
    print(f"  interaction: coef_pullback={it['coef_drawdown_pullback']:+.4f} "
          f"coef_crash={it['coef_drawdown_crash']:+.4f} "
          f"p_interaction={it['p_interaction']:.4f}")


if __name__ == "__main__":
    main()
