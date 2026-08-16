"""H1 floor effect test: distribución de PE_min absoluta por severidad de drawdown."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import (
    CHARTS, MIN_EVENT_DAYS_MAIN, OUTPUTS, PE_MAIN,
    compute_event_metrics, detect_events, load_price_ret_pe,
)


BUCKETS = [
    ("5-15%", 0.05, 0.15),
    ("15-25%", 0.15, 0.25),
    ("25-40%", 0.25, 0.40),
    (">40%", 0.40, 1.00),
]


def enrich_with_pe_min(events, pe, min_event_days):
    rows = []
    for _, ev in events.iterrows():
        peak, trough = ev["peak_date"], ev["trough_date"]
        if (trough - peak).days < min_event_days:
            continue
        pe_during = pe.loc[peak:trough]
        if len(pe_during) < 5:
            continue
        pe_before = pe.loc[:peak]
        if len(pe_before) < 60:
            continue
        rows.append({
            "peak_date": peak,
            "trough_date": trough,
            "drawdown": ev["drawdown"],
            "pe_pre_peak": pe_before.iloc[-60:].mean(),
            "pe_min_during": pe_during.min(),
            "pe_mean_during": pe_during.mean(),
        })
    return pd.DataFrame(rows)


def bucket_events(df):
    df = df.copy()
    df["abs_dd"] = df["drawdown"].abs()
    df["bucket"] = "N/A"
    for name, lo, hi in BUCKETS:
        mask = (df["abs_dd"] > lo) & (df["abs_dd"] <= hi)
        df.loc[mask, "bucket"] = name
    return df


def main():
    close, log_ret, pe = load_price_ret_pe(pe_window=PE_MAIN["window"], m=PE_MAIN["m"])
    events = detect_events(close)
    df = enrich_with_pe_min(events, pe, MIN_EVENT_DAYS_MAIN)
    df = bucket_events(df)

    summary_rows = []
    for name, _, _ in BUCKETS:
        sub = df[df["bucket"] == name]
        if len(sub) == 0:
            continue
        summary_rows.append({
            "bucket": name,
            "n": len(sub),
            "pe_min_mean": sub["pe_min_during"].mean(),
            "pe_min_std": sub["pe_min_during"].std(),
            "pe_min_median": sub["pe_min_during"].median(),
            "pe_min_min": sub["pe_min_during"].min(),
            "pe_min_max": sub["pe_min_during"].max(),
            "pe_pre_mean": sub["pe_pre_peak"].mean(),
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(OUTPUTS / "e3_h1_floor_summary.csv", index=False)

    # Plot 1: boxplot PE_min by bucket
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df[df["bucket"] == name]["pe_min_during"].values
             for name, _, _ in BUCKETS]
    labels = [f"{name}\n(n={len(d)})" for (name, _, _), d in zip(BUCKETS, data)]
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("PE mínima durante evento (unidades absolutas)")
    ax.set_xlabel("Severidad del drawdown")
    ax.set_title("Distribución de PE mínima durante cada evento por severidad")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHARTS / "e3_h1_floor_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: scatter drawdown vs PE_min absoluta
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"5-15%": "steelblue", "15-25%": "orange",
              "25-40%": "crimson", ">40%": "darkred"}
    for name, _, _ in BUCKETS:
        sub = df[df["bucket"] == name]
        if len(sub) == 0:
            continue
        ax.scatter(sub["abs_dd"] * 100, sub["pe_min_during"],
                    c=colors[name], label=f"{name} (n={len(sub)})",
                    s=48, alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Caída de mercado ($|$drawdown$|$, \\%)")
    ax.set_ylabel("PE mínima observada durante el evento")
    ax.set_title("Piso de PE por severidad del drawdown")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(CHARTS / "e3_h1_floor_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote e3_h1_floor_summary.csv, e3_h1_floor_boxplot.png, e3_h1_floor_scatter.png")


if __name__ == "__main__":
    main()
