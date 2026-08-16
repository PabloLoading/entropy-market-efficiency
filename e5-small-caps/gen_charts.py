"""E5 charts for the thesis: rolling PE series with crisis shading + annual gap."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

WINDOWS = [252, 504]

CRISES = [
    ("2000-03-24", "2002-10-09", "Dot-com"),
    ("2007-10-09", "2009-03-09", "Crisis 2008"),
    ("2020-02-19", "2020-03-23", "COVID"),
]


def chart_series() -> None:
    fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(12, 7), sharex=True)
    for ax, w in zip(axes, WINDOWS):
        df = pd.read_csv(OUT_DIR / f"pe_series_w{w}.csv", index_col="date",
                         parse_dates=True)
        ax.plot(df.index, df["pe_sp600"], label="S&P 600 (small caps)",
                color="C1", linewidth=0.9)
        ax.plot(df.index, df["pe_sp500"], label="S&P 500 (large caps)",
                color="C0", linewidth=0.9)
        for start, end, name in CRISES:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       color="gray", alpha=0.2)
            if w == WINDOWS[0]:
                ax.text(pd.Timestamp(start), ax.get_ylim()[0] + 0.001, name,
                        fontsize=7, color="gray", rotation=90, va="bottom")
        ax.set_ylabel(f"PE normalizada (W={w})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("Fecha")
    fig.suptitle("PE rolling: S&P 600 vs S&P 500 (m=3, diario)")
    plt.tight_layout()
    path = OUT_DIR / "e5_pe_series.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"-> {path.name}")


def chart_annual_gap() -> None:
    fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(12, 6), sharex=True)
    for ax, w in zip(axes, WINDOWS):
        annual = pd.read_csv(OUT_DIR / f"annual_gap_w{w}.csv", index_col="year")
        colors = ["C1" if v >= 0 else "C0" for v in annual["mean"]]
        ax.bar(annual.index, annual["mean"], yerr=annual["std"], color=colors,
               alpha=0.8, edgecolor="gray", error_kw={"alpha": 0.4})
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"Brecha PE (W={w})")
        ax.grid(True, alpha=0.3, axis="y")
    axes[-1].set_xlabel("Año")
    fig.suptitle("Brecha anual de PE: S&P 600 menos S&P 500\n"
                 "(positivo = small caps con mayor entropía)")
    plt.tight_layout()
    path = OUT_DIR / "e5_brecha_anual.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"-> {path.name}")


def print_summary_table() -> None:
    with open(OUT_DIR / "results.json") as f:
        res = json.load(f)["results"]
    print("\nResumen (diferencia S&P 600 menos S&P 500):")
    print(f"{'W':>5} {'PE 600':>8} {'PE 500':>8} {'diff':>9} {'CI 95%':>22} {'n':>4}")
    for key, r in res.items():
        w = key[1:]
        ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        print(f"{w:>5} {r['pe_sp600_mean']:>8.4f} {r['pe_sp500_mean']:>8.4f} "
              f"{r['diff_mean_nonoverlap']:>+9.4f} {ci:>22} {r['n_nonoverlap']:>4}")


def main() -> None:
    chart_series()
    chart_annual_gap()
    print_summary_table()


if __name__ == "__main__":
    main()
