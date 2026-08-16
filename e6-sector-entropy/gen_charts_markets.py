"""E6 sub-experimento 2 charts: choropleth mundial de eficiencia, crisis por
mercado, y tabla resumen impresa."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

ISO3 = {"us": "USA", "japon": "JPN", "uk": "GBR", "hongkong": "HKG",
        "alemania": "DEU", "australia": "AUS", "brasil": "BRA",
        "china": "CHN", "india": "IND"}
NAMES = {"us": "Estados Unidos", "japon": "Japón", "uk": "Reino Unido",
         "hongkong": "Hong Kong", "alemania": "Alemania", "australia": "Australia",
         "brasil": "Brasil", "china": "China", "india": "India"}
CRISES_ORDER = ["dot-com", "crisis-2008", "covid", "tasas-2022", "china-2015 (local)"]
CRISES_TITLES = ["Dot-com", "Crisis 2008", "COVID", "Tasas 2022", "China 2015 (local)"]


def chart_choropleth() -> None:
    with open(OUT / "markets_percentiles.json") as f:
        percs = json.load(f)["w252"]
    rows = [{"iso": ISO3[k], "mercado": r["sector"],
             "percentil": r["percentil_promedio"]} for k, r in percs.items()]
    df = pd.DataFrame(rows)
    fig = px.choropleth(df, locations="iso", color="percentil",
                        hover_name="mercado", color_continuous_scale="RdYlGn",
                        range_color=(0.3, 0.7),
                        labels={"percentil": "Percentil promedio de PE"})
    fig.update_layout(
        title="Eficiencia informacional por mercado (percentil promedio del ranking anual de PE, 2000-2025)",
        margin=dict(l=10, r=10, t=60, b=10), width=1100, height=600,
        coloraxis_colorbar=dict(title="Percentil<br>promedio"))
    fig.write_image(OUT / "e6_choropleth.png", scale=2)
    print("-> e6_choropleth.png")


def chart_markets_crisis() -> None:
    cr = pd.read_csv(OUT / "markets_crisis.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, key, title in zip(axes.flat, CRISES_ORDER, CRISES_TITLES):
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
        ax2.set_xlabel("Drawdown (%)", fontsize=8, color="C3")
        ax.tick_params(axis="x", labelsize=7, colors="C0")
        ax2.tick_params(axis="x", labelsize=7, colors="C3")
        ax.grid(True, alpha=0.3, axis="x")
    axes.flat[-1].axis("off")
    fig.suptitle("Caída de PE vs drawdown de precio por mercado en cada crisis")
    plt.tight_layout()
    plt.savefig(OUT / "e6_markets_crisis.png", dpi=150)
    plt.close()
    print("-> e6_markets_crisis.png")


def print_summary() -> None:
    with open(OUT / "markets_percentiles.json") as f:
        percs = json.load(f)["w252"]
    print("\nTabla resumen mercados (W=252):")
    print(f"{'mercado':16} {'perc':>6} {'CI 95%':>17} {'conf':>7} {'PE m6 est.':>10} {'tendencia MK':>22}")
    for k, r in sorted(percs.items(), key=lambda kv: -kv[1]["percentil_promedio"]):
        mk = r["mann_kendall"]
        print(f"{r['sector']:16} {r['percentil_promedio']:>6.3f} "
              f"[{r['ci95'][0]:>6.3f},{r['ci95'][1]:>6.3f}] "
              f"{r['confianza_direccional']:>6.1%} {r.get('pe_estatica_m6', float('nan')):>10.4f} "
              f"{mk['tendencia']:>15} (p={mk['p']})")


def main() -> None:
    chart_choropleth()
    chart_markets_crisis()
    print_summary()


if __name__ == "__main__":
    main()
