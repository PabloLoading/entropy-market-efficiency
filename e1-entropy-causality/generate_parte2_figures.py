"""Generate the heatmap for Parte II Hallazgos: assets x factors with % of
rolling windows where each factor causes FutureReturn robustly."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parte2_helpers import PARTE2_CHARTS


CACHE_DIR = Path("outputs/.cache/parte2")

# Asset groups for row ordering
GROUPS = {
    "Sectores": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLB", "XLU", "XLP", "XLY"],
    "Índices": ["SPY", "QQQ", "IWM"],
    "Acciones": ["TSLA", "NVDA", "NFLX", "MELI", "KO", "WMT", "MSFT", "INTC"],
}
FACTORS = ["Returns", "Volatility", "Liquidity", "Entropy"]


def collect():
    rows = []
    for pkl in CACHE_DIR.glob("daily_*.pkl"):
        parts = pkl.stem.split("_")
        asset = parts[1]
        with open(pkl, "rb") as f:
            results = pickle.load(f)
        for r in results:
            labels = r["labels"]
            if "FutureReturn" not in labels:
                continue
            fr_idx = labels.index("FutureReturn")
            for src in FACTORS:
                if src not in labels:
                    continue
                s_idx = labels.index(src)
                p = float(r["edge_prob"][fr_idx, s_idx])
                rows.append({"asset": asset, "factor": src, "is_robust": p >= 0.5})
    return pd.DataFrame(rows)


def main():
    df = collect()
    # Percentage of robust windows per (asset, factor)
    tab = df.groupby(["asset", "factor"])["is_robust"].mean().unstack() * 100

    # Order rows by group
    ordered = []
    row_labels = []
    for group, assets in GROUPS.items():
        for a in assets:
            if a in tab.index:
                ordered.append(a)
                row_labels.append(a)
    M = tab.reindex(ordered)[FACTORS]

    # Plot
    fig, ax = plt.subplots(figsize=(6.2, 8.6))
    im = ax.imshow(M.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=8)

    ax.set_xticks(range(len(FACTORS)))
    ax.set_xticklabels(FACTORS)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Cell values
    for i in range(len(row_labels)):
        for j in range(len(FACTORS)):
            v = M.values[i, j]
            color = "white" if v > 4 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color=color, fontsize=8)

    # Group separators
    group_ends = []
    cum = 0
    for group, assets in GROUPS.items():
        cum += sum(1 for a in assets if a in tab.index)
        group_ends.append(cum)
    for e in group_ends[:-1]:
        ax.axhline(e - 0.5, color="black", linewidth=1.2)

    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("% ventanas rolling con edge robusto hacia FutureReturn",
                    fontsize=9)

    ax.set_title("Frecuencia de aristas causales por activo y factor\n"
                 "(rolling 200d + 400d, umbral $\\geq 0.5$)", fontsize=11)
    plt.tight_layout()
    out = PARTE2_CHARTS / "parte2_heatmap_sector_factor.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def hourly_heatmap():
    HOURLY_ASSETS_ORDER = ["SPY", "QQQ", "IWM", "XLU", "XLF", "XLK", "XLE", "TSLA"]
    rows = []
    for pkl in CACHE_DIR.glob("hourly_*.pkl"):
        asset = pkl.stem.split("_")[1]
        with open(pkl, "rb") as f:
            results = pickle.load(f)
        for r in results:
            labels = r["labels"]
            if "FutureReturn" not in labels:
                continue
            fr = labels.index("FutureReturn")
            for src in FACTORS:
                if src not in labels:
                    continue
                s = labels.index(src)
                rows.append({"asset": asset, "factor": src,
                             "edge_prob": float(r["edge_prob"][fr, s])})
    df = pd.DataFrame(rows)
    tab = df.groupby(["asset", "factor"])["edge_prob"].max().unstack()
    M = tab.reindex(HOURLY_ASSETS_ORDER)[FACTORS]

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(M.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(FACTORS)))
    ax.set_xticklabels(FACTORS)
    ax.set_yticks(range(len(HOURLY_ASSETS_ORDER)))
    ax.set_yticklabels(HOURLY_ASSETS_ORDER, fontsize=9)

    for i in range(len(HOURLY_ASSETS_ORDER)):
        for j in range(len(FACTORS)):
            v = M.values[i, j]
            color = "white" if v > 0.6 else "black"
            weight = "bold" if v >= 0.5 else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8, fontweight=weight)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Máx. edge\\_prob al hourly", fontsize=9)

    ax.set_title("Máxima probabilidad de arista causal hacia FutureReturn\n"
                 "por activo y factor (rolling 100d, hourly)", fontsize=10)
    plt.tight_layout()
    out = PARTE2_CHARTS / "parte2_heatmap_hourly.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
    hourly_heatmap()
