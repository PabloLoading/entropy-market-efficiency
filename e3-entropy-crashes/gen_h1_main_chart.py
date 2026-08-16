"""Generate main H1 scatter chart: W=140, cutoff 25%."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from helpers import (
    CHARTS, MIN_EVENT_DAYS_MAIN, classify, compute_event_metrics,
    detect_events, load_price_ret_pe,
)

W = 140
CUTOFF = 0.25


def fit(df):
    X = sm.add_constant(df[["abs_drawdown"]])
    return sm.OLS(df["delta_pe"], X).fit(cov_type="HC3")


def main():
    close, log_ret, pe = load_price_ret_pe(pe_window=W, m=3)
    events = detect_events(close)
    df = compute_event_metrics(events, log_ret, pe, pe_window=W,
                                min_event_days=MIN_EVENT_DAYS_MAIN)
    df["type"] = classify(df["drawdown"], cutoff=0.15)
    df["abs_drawdown"] = df["drawdown"].abs()
    df = df[df["abs_drawdown"] <= CUTOFF].copy()

    m_all = fit(df)
    m_pull = fit(df[df["type"] == "pullback"])
    m_crash = fit(df[df["type"] == "crash"])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = {"pullback": "steelblue", "crash": "crimson"}
    for cat in ["pullback", "crash"]:
        sub = df[df["type"] == cat]
        ax.scatter(sub["abs_drawdown"] * 100, sub["delta_pe"],
                    c=colors[cat], label=f"{cat} (n={len(sub)})", s=48,
                    alpha=0.8, edgecolor="black", linewidth=0.4)

    for name, model, sub, color in [
        ("pullback", m_pull, df[df["type"] == "pullback"], "steelblue"),
        ("crash", m_crash, df[df["type"] == "crash"], "crimson"),
    ]:
        xr = np.linspace(sub["abs_drawdown"].min(), sub["abs_drawdown"].max(), 50)
        yr = model.params["const"] + model.params["abs_drawdown"] * xr
        p = model.pvalues["abs_drawdown"]
        ax.plot(xr * 100, yr, "--", color=color, lw=1.4, alpha=0.9,
                 label=f"OLS {name}: $\\beta={model.params['abs_drawdown']:+.3f}$, $p={p:.3f}$")

    xr = np.linspace(df["abs_drawdown"].min(), df["abs_drawdown"].max(), 100)
    yr = m_all.params["const"] + m_all.params["abs_drawdown"] * xr
    ax.plot(xr * 100, yr, "k--", lw=1.6,
             label=f"OLS todos: $\\beta={m_all.params['abs_drawdown']:+.3f}$, "
                   f"$p<0{{,}}001$, $R^2={m_all.rsquared:.2f}$")

    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Caída de mercado ($|$drawdown$|$, \\%)")
    ax.set_ylabel("Caída de PE (absoluta)")
    ax.set_title(f"H1: relación drawdown vs. caída de PE ($W={W}$, $|$dd$| \\leq 25\\%$)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHARTS / "e3_h1_main.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved e3_h1_main.png, n={len(df)}")


if __name__ == "__main__":
    main()
