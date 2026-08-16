"""E6 charts sub-experimento 1: series con crisis, ranking/rotacion, niveles con CI,
matriz de co-evolucion."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

SECTORS = {
    "XLK": "Tecnología", "XLE": "Energía", "XLV": "Salud", "XLF": "Financiero",
    "XLP": "Consumo básico", "XLY": "Consumo discrecional", "XLI": "Industrial",
    "XLU": "Utilities", "XLB": "Materiales", "XLRE": "Real Estate",
    "XLC": "Comunicaciones",
}
CRISES = [
    ("2000-03-24", "2002-10-09", "Dot-com"),
    ("2007-10-09", "2009-03-09", "Crisis 2008"),
    ("2020-02-19", "2020-03-23", "COVID"),
    ("2022-01-03", "2022-10-12", "Tasas 2022"),
]


def chart_series() -> None:
    df = pd.read_csv(OUT / "pe_series_w252.csv", index_col="date", parse_dates=True)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for t in SECTORS:
        if t in df.columns:
            ax.plot(df.index, df[t], linewidth=0.7, alpha=0.45)
    ax.plot(df.index, df["SPY"], color="black", linewidth=1.4,
            label="S&P 500 (referencia)")
    for s, e, name in CRISES:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="gray", alpha=0.18)
        ax.text(pd.Timestamp(s), ax.get_ylim()[0] + 0.002, name, fontsize=7,
                color="gray", rotation=90, va="bottom")
    ax.set_ylabel("PE normalizada (W=252)")
    ax.set_xlabel("Fecha")
    ax.set_title("PE rolling de los 11 sectores (líneas finas) y el S&P 500")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "e6_pe_series.png", dpi=150)
    plt.close()
    print("-> e6_pe_series.png")


def chart_rotation() -> None:
    with open(OUT / "rotacion.json") as f:
        rot = json.load(f)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   height_ratios=[1, 1.4])
    # Panel 1: Spearman entre rankings de años consecutivos
    pairs = rot["spearman_detalle"]
    years = [int(p["years"].split("-")[1]) for p in pairs]
    rhos = [p["rho"] for p in pairs]
    ax1.bar(years, rhos, color="C0", alpha=0.85)
    ax1.axhline(rot["spearman_consecutivo_promedio"], color="black",
                linestyle="--", linewidth=1,
                label=f"Promedio ({rot['spearman_consecutivo_promedio']:.2f})")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_ylabel("Spearman vs año anterior")
    ax1.set_title("Estabilidad del ranking sectorial de PE (W=252)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    # Panel 2: qué sector fue el menos eficiente cada año
    last = rot["ultimo_puesto_por_anio"]
    order = [SECTORS[t] for t in SECTORS]
    ax2.scatter([int(y) for y in last], [order.index(s) for s in last.values()],
                s=60, color="C3")
    ax2.set_yticks(range(len(order)), order, fontsize=8)
    ax2.set_ylabel("Sector con menor PE del año")
    ax2.set_xlabel("Año")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "e6_rotacion.png", dpi=150)
    plt.close()
    print("-> e6_rotacion.png")


def chart_levels() -> None:
    with open(OUT / "niveles.json") as f:
        lev = json.load(f)
    rows = [(r["sector"], r["pe_w252"], r["ci_w252"]) for t, r in lev.items()
            if t in SECTORS and "pe_w252" in r]
    rows.sort(key=lambda x: x[1])
    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    los = [r[1] - r[2][0] for r in rows]
    his = [r[2][1] - r[1] for r in rows]
    spy = lev["SPY"]["pe_w252"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(means, range(len(rows)), xerr=[los, his], fmt="o", color="C0",
                capsize=3)
    ax.axvline(spy, color="black", linestyle="--", linewidth=1,
               label=f"S&P 500 ({spy:.4f})")
    ax.set_yticks(range(len(rows)), names)
    ax.set_xlabel("PE media 1999-2025 (W=252), CI 95%")
    ax.set_title("Niveles de PE por sector")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUT / "e6_niveles.png", dpi=150)
    plt.close()
    print("-> e6_niveles.png")


def chart_crises() -> None:
    cr = pd.read_csv(OUT / "crisis.csv")
    cr = cr[cr["ticker"] != "SPY"]
    episodes = [c[2] for c in CRISES]
    keys = ["dot-com", "crisis-2008", "covid", "tasas-2022"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    for ax, key, title in zip(axes.flat, keys, episodes):
        sub = cr[cr["episodio"] == key].sort_values("caida", ascending=True)
        y = np.arange(len(sub))
        ax.barh(y + 0.2, sub["caida"], height=0.38, color="C0", alpha=0.9,
                label="Caída de PE (W=140)")
        ax2 = ax.twiny()
        ax2.barh(y - 0.2, -sub["drawdown_precio"] * 100, height=0.38, color="C3",
                 alpha=0.55, label="Drawdown de precio (%)")
        ax.set_yticks(y, sub["sector"], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Caída de PE", fontsize=8, color="C0")
        ax2.set_xlabel("Drawdown de precio (%)", fontsize=8, color="C3")
        ax.tick_params(axis="x", labelsize=7, colors="C0")
        ax2.tick_params(axis="x", labelsize=7, colors="C3")
        ax.grid(True, alpha=0.3, axis="x")
    fig.suptitle("Caída de PE vs drawdown de precio por sector en cada crisis\n"
                 "(las dos magnitudes no correlacionan: la PE cae en el foco del "
                 "shock, no donde más cae el precio)")
    plt.tight_layout()
    plt.savefig(OUT / "e6_crisis.png", dpi=150)
    plt.close()
    print("-> e6_crisis.png")


def chart_sincronia() -> None:
    """Fechas del minimo de PE por sector en cada episodio: columna vertical =
    sincronizado, escalera = escalonado. Sombreado: ventana pico-valle del crash."""
    cr = pd.read_csv(OUT / "crisis.csv", parse_dates=["fecha_min"])
    keys = ["dot-com", "crisis-2008", "covid", "tasas-2022"]
    titles = [c[2] for c in CRISES]
    spans = {k: (pd.Timestamp(s), pd.Timestamp(e)) for k, (s, e, _) in
             zip(keys, [(c[0], c[1], c[2]) for c in CRISES])}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    for ax, key, title in zip(axes.flat, keys, titles):
        sub = cr[cr["episodio"] == key].sort_values("fecha_min").reset_index(drop=True)
        start, end = spans[key]
        ax.axvspan(start, end, color="gray", alpha=0.15,
                   label="Crash (pico a valle del S&P 500)")
        for i, row in sub.iterrows():
            is_spy = row["ticker"] == "SPY"
            ax.scatter(row["fecha_min"], i, s=max(row["caida"], 0) * 12000 + 15,
                       color="black" if is_spy else "C0", zorder=3)
        ax.set_yticks(range(len(sub)),
                      [r["sector"] if r["ticker"] != "SPY" else "S&P 500"
                       for _, r in sub.iterrows()], fontsize=8)
        solo_sec = sub[sub["ticker"] != "SPY"]["fecha_min"]
        rango = (solo_sec.max() - solo_sec.min()).days
        ax.set_title(f"{title} (rango de mínimos: {rango} días)", fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")
        ax.tick_params(axis="x", labelsize=8)
        if key == "dot-com":
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Fecha del mínimo de PE por sector en cada crisis (W=140)\n"
                 "Tamaño del punto = magnitud de la caída de PE. Columna vertical = "
                 "sincronizado; escalera = escalonado.")
    plt.tight_layout()
    plt.savefig(OUT / "e6_sincronia.png", dpi=150)
    plt.close()
    print("-> e6_sincronia.png")


def chart_coevolution() -> None:
    corr = pd.read_csv(OUT / "coevolucion.csv", index_col=0)
    labels = [SECTORS[t] for t in corr.columns]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, shrink=0.8, label="Correlación de cambios mensuales de PE")
    ax.set_title("Co-evolución de la PE entre sectores (W=252)")
    plt.tight_layout()
    plt.savefig(OUT / "e6_coevolucion.png", dpi=150)
    plt.close()
    print("-> e6_coevolucion.png")


def main() -> None:
    chart_series()
    chart_rotation()
    chart_levels()
    chart_crises()
    chart_sincronia()
    chart_coevolution()


if __name__ == "__main__":
    main()
