"""E7 charts: perfil horario (m=3 y m=4) con CIs, curva intradia de 5 minutos,
y evolucion del gap apertura-mediodia."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def chart_perfil() -> None:
    r = json.load(open(OUT / "intraday_results.json"))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, cfg, title in [(axes[0], "horaria_m3", "m=3 (patrones de 3 barras de 5 min)"),
                            (axes[1], "horaria_m4", "m=4 (patrones de 4 barras de 5 min)")]:
        fr = r[cfg]["franjas"]
        x = np.arange(len(fr))
        pe = [f["pe"] for f in fr]
        lo = [f["pe"] - f["ci95"][0] for f in fr]
        hi = [f["ci95"][1] - f["pe"] for f in fr]
        ax.errorbar(x, pe, yerr=[lo, hi], fmt="o-", capsize=4, color="C0")
        ax.set_xticks(x, [f["label"] for f in fr])
        ax.set_ylabel("PE normalizada")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Franja horaria (inicio, hora del Este)")
    fig.suptitle("Perfil horario de PE del S&P 500, 2008-2021 (CI 95% por bootstrap de días)")
    plt.tight_layout()
    plt.savefig(OUT / "e7_perfil.png", dpi=150)
    plt.close()
    print("-> e7_perfil.png")


def chart_curva() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, fname, title in [
            (axes[0], "curve.csv", "m=3 (patrones de 3 barras de 1 min)"),
            (axes[1], "curve_m4.csv", "m=4 (patrones de 4 barras de 1 min)")]:
        c = pd.read_csv(OUT / fname)
        x = np.arange(len(c))
        ax.plot(x, c["pe"], color="C0", linewidth=1.2)
        ax.fill_between(x, c["lo"], c["hi"], color="C0", alpha=0.2, label="CI 95%")
        ax.annotate("apertura", xy=(0, c["pe"].iloc[0]), xytext=(3, c["pe"].min()),
                    arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8,
                    color="gray")
        ax.annotate("últimos 5 min", xy=(len(c) - 1, c["pe"].iloc[-1]),
                    xytext=(len(c) - 14, c["pe"].min()),
                    arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8,
                    color="gray")
        ax.set_ylabel("PE normalizada")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    step = 6
    ax.set_xticks(x[::step], c["time"][::step], rotation=45, fontsize=7)
    axes[1].set_xlabel("Grupo de 5 minutos (hora del Este)")
    fig.suptitle("Perfil intradía de PE: grupos de 5 minutos sobre barras de 1 minuto, 2008-2021")
    plt.tight_layout()
    plt.savefig(OUT / "e7_curva.png", dpi=150)
    plt.close()
    print("-> e7_curva.png")


def chart_evolucion() -> None:
    g = json.load(open(OUT / "evolution_gap.json"))
    years = [int(y) for y in g]
    gap = [g[str(y)]["gap"] for y in years]
    lo = [g[str(y)]["gap"] - g[str(y)]["ci95"][0] for y in years]
    hi = [g[str(y)]["ci95"][1] - g[str(y)]["gap"] for y in years]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.errorbar(years, gap, yerr=[lo, hi], fmt="o", capsize=4, color="C0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("PE apertura − PE mediodía")
    ax.set_xlabel("Año")
    ax.set_title("Gap apertura-mediodía por año (CI 95%): positivo = apertura más aleatoria")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "e7_evolucion.png", dpi=150)
    plt.close()
    print("-> e7_evolucion.png")


def main() -> None:
    chart_perfil()
    chart_curva()
    chart_evolucion()


if __name__ == "__main__":
    main()
