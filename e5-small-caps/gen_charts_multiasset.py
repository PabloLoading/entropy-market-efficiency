"""E5 fase 2 charts: distribuciones de PE por accion, brecha anual en dos niveles,
PE vs dollar volume, y tabla final combinada (indices + acciones)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

LABELS = {"sp600": "S&P 600 (small caps)", "sp500": "S&P 500 (large caps)"}
COLORS = {"sp600": "C1", "sp500": "C0"}


def load_valid() -> pd.DataFrame:
    sy = pd.read_parquet(OUT_DIR / "stock_year_pe.parquet")
    return sy[(sy["excluded"] == "") & sy["pe"].notna()].copy()


def chart_distributions(valid: pd.DataFrame) -> None:
    sub = valid[valid["kind"] == "anual"]
    per_stock = sub.groupby(["ticker", "group"], as_index=False)["pe"].mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    xlo = 0.985
    n_out = (per_stock["pe"] < xlo).sum()
    for grp in ["sp500", "sp600"]:
        vals = per_stock.loc[per_stock["group"] == grp, "pe"].values
        ax1.hist(vals, bins=np.linspace(xlo, 1.0, 45), alpha=0.55,
                 label=LABELS[grp], color=COLORS[grp], density=True)
        xs = np.sort(vals)
        ax2.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=COLORS[grp],
                 label=LABELS[grp])
        ax2.axvline(np.median(vals), color=COLORS[grp], linestyle=":", alpha=0.7)
    ax1.set_xlim(xlo, 1.0)
    ax2.set_xlim(xlo, 1.0)
    if n_out:
        ax1.text(0.02, 0.95, f"{n_out} acciones < {xlo} fuera de rango\n"
                 "(ties altos, ver write-up)", transform=ax1.transAxes,
                 fontsize=7, va="top", color="gray")
    ax1.set_xlabel("PE media por acción (W=252)")
    ax1.set_ylabel("Densidad")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("PE media por acción (W=252)")
    ax2.set_ylabel("CDF")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Distribución de PE por acción: S&P 600 vs S&P 500 (2015-2025)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "e5_ma_distribuciones.png", dpi=150)
    plt.close()
    print("-> e5_ma_distribuciones.png")


def chart_annual_gap(valid: pd.DataFrame) -> None:
    sub = valid[valid["kind"] == "anual"]
    stock_gap = (sub.groupby(["period", "group"])["pe"].mean().unstack()
                 .assign(gap=lambda d: d["sp600"] - d["sp500"]))
    idx = pd.read_csv(OUT_DIR / "annual_gap_w252.csv", index_col="year")
    idx = idx.loc[idx.index >= 2015, "mean"]
    years = stock_gap.index.astype(int)
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.2, stock_gap["gap"].values, width=0.4,
           label="Nivel acción (media por grupo)", color="C3", alpha=0.8)
    ax.bar(x + 0.2, idx.reindex(years).values, width=0.4,
           label="Nivel índice (fase 1)", color="C7", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, years)
    ax.set_ylabel("Brecha PE (S&P 600 − S&P 500)")
    ax.set_xlabel("Año")
    ax.set_title("Brecha anual de PE en dos niveles de agregación (W=252)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "e5_ma_brecha_anual.png", dpi=150)
    plt.close()
    print("-> e5_ma_brecha_anual.png")


def chart_pe_vs_dv(valid: pd.DataFrame) -> None:
    sub = valid[valid["kind"] == "anual"]
    per_stock = sub.groupby(["ticker", "group"], as_index=False).agg(
        pe=("pe", "mean"), dv=("dollar_volume", "median"))
    per_stock = per_stock[(per_stock["dv"] > 1e5) & (per_stock["pe"] > 0.985)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for grp in ["sp500", "sp600"]:
        d = per_stock[per_stock["group"] == grp]
        ax.scatter(np.log10(d["dv"]), d["pe"], s=8, alpha=0.4,
                   color=COLORS[grp], label=LABELS[grp])
    rho = per_stock["pe"].corr(np.log(per_stock["dv"]), method="spearman")
    ax.set_xlabel("log10(dollar volume mediano diario, USD)")
    ax.set_ylabel("PE media por acción (W=252)")
    ax.set_title(f"PE vs tamaño/liquidez por acción (Spearman = {rho:+.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "e5_ma_pe_vs_dv.png", dpi=150)
    plt.close()
    print("-> e5_ma_pe_vs_dv.png")


def chart_tail_deciles(valid: pd.DataFrame) -> None:
    """Share of small caps per PE decile: the tail effect chart."""
    sub = valid[valid["kind"] == "anual"]
    ties_avg = sub.groupby("ticker")["ties"].mean()
    sub = sub[sub["ticker"].isin(ties_avg[ties_avg < 0.02].index)]
    ps = sub.groupby(["ticker", "group"], as_index=False).agg(
        pe=("pe", "mean"), n_years=("pe", "count"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = ps.copy()
    d["decil"] = pd.qcut(d["pe"], 10, labels=range(1, 11))
    share = d.groupby("decil", observed=True)["group"].apply(
        lambda g: (g == "sp600").mean())
    base = (d["group"] == "sp600").mean()
    ax.bar(share.index.astype(str), share.values * 100, color="C0", alpha=0.85,
           edgecolor="gray")
    ax.axhline(base * 100, color="black", linestyle="--", linewidth=1.2,
               label=f"Peso de las small caps en la muestra ({base:.0%})")
    ax.set_xlabel("Decil de PE por acción (1 = menor PE)")
    ax.set_ylabel("% de small caps en el decil")
    ax.set_title(f"Composición de los deciles de PE (W=252, n={len(d)} acciones)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = OUT_DIR / "e5_ma_cola_deciles.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"-> {path.name}")


def print_combined_table() -> None:
    with open(OUT_DIR / "results.json") as f:
        idx = json.load(f)["results"]
    with open(OUT_DIR / "multiasset_results.json") as f:
        ma = json.load(f)["configs"]
    print("\nTabla final combinada (diff = S&P 600 menos S&P 500):")
    print(f"{'Nivel':>8} {'W':>5} {'PE 600':>8} {'PE 500':>8} {'diff':>9} {'CI 95%':>22}")
    for key, r in idx.items():
        ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        print(f"{'índice':>8} {key[1:]:>5} {r['pe_sp600_mean']:>8.4f} "
              f"{r['pe_sp500_mean']:>8.4f} {r['diff_mean_nonoverlap']:>+9.4f} {ci:>22}")
    for key, r in ma.items():
        ci = f"[{r['diff_mean_ci95'][0]:+.4f}, {r['diff_mean_ci95'][1]:+.4f}]"
        print(f"{'acción':>8} {key[1:]:>5} {r['pe_mean']['sp600']:>8.4f} "
              f"{r['pe_mean']['sp500']:>8.4f} {r['diff_mean']:>+9.4f} {ci:>22}")


def main() -> None:
    valid = load_valid()
    chart_distributions(valid)
    chart_annual_gap(valid)
    chart_pe_vs_dv(valid)
    chart_tail_deciles(valid)
    print_combined_table()


if __name__ == "__main__":
    main()
