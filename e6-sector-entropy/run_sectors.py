"""E6 sub-experimento 1: PE rolling por sector y los 4 análisis pre-declarados.
Usa la maquinaria compartida de e6lib (misma que run_markets.py).

Produce en outputs/:
- pe_series_w{140,252,504}.csv  series rolling de PE por ticker
- niveles.json / niveles.csv    PE media por sector con CI 95% + puesto promedio
- rankings_anuales*.csv         percentil de cada sector por año (2000-2025, 3 ventanas)
- rotacion.json                 Spearman entre años consecutivos + permanencia del último puesto
- percentiles.json              percentil promedio con CI y confianza (3 ventanas)
- crisis.csv / crisis_stats.json  caídas de PE vs drawdown por episodio (W=140)
- coevolucion.csv               matriz de correlaciones de cambios mensuales de PE (W=252)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import e6lib

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"

SECTORS = {
    "XLK": "Tecnología", "XLE": "Energía", "XLV": "Salud", "XLF": "Financiero",
    "XLP": "Consumo básico", "XLY": "Consumo discrecional", "XLI": "Industrial",
    "XLU": "Utilities", "XLB": "Materiales", "XLRE": "Real Estate",
    "XLC": "Comunicaciones",
}
REFERENCE = "SPY"
ALL_TICKERS = list(SECTORS) + [REFERENCE]
ALL_NAMES = {**SECTORS, REFERENCE: "S&P 500 (referencia)"}

WINDOWS = [140, 252, 504]
RANK_YEARS = range(2000, 2026)
BOOT_REPS = 2000
SEED = 42

CRISES = {
    "dot-com": ("2000-03-24", "2002-10-09"),
    "crisis-2008": ("2007-10-09", "2009-03-09"),
    "covid": ("2020-02-19", "2020-03-23"),
    "tasas-2022": ("2022-01-03", "2022-10-12"),
}


def compute_series() -> dict[int, pd.DataFrame]:
    if all((OUT / f"pe_series_w{w}.csv").exists() for w in WINDOWS):
        print("  (series cacheadas en outputs/, salteando computo)")
        return {w: pd.read_csv(OUT / f"pe_series_w{w}.csv", index_col="date",
                               parse_dates=True) for w in WINDOWS}
    series: dict[int, dict[str, pd.Series]] = {w: {} for w in WINDOWS}
    for t in ALL_TICKERS:
        close = pd.read_parquet(DATA / f"{t}.parquet")["Adj Close"].dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        ret = np.log(close / close.shift(1)).dropna()
        for w in WINDOWS:
            if len(ret) > w:
                series[w][t] = e6lib.rolling_pe(ret, w)
        print(f"  {t}: PE rolling listo ({len(ret)} retornos)")
    frames = {}
    for w in WINDOWS:
        df = pd.DataFrame(series[w]).dropna(how="all")
        df.to_csv(OUT / f"pe_series_w{w}.csv", index_label="date")
        frames[w] = df
    return frames


def analyze_levels(pe252: pd.DataFrame, pe504: pd.DataFrame,
                   rankings: pd.DataFrame) -> dict:
    res = {}
    for t in ALL_TICKERS:
        row: dict = {"sector": ALL_NAMES[t]}
        for w, df in [(252, pe252), (504, pe504)]:
            if t not in df.columns:
                continue
            s = df[t].dropna()
            nov = s.iloc[::w]
            lo, hi = e6lib.block_bootstrap_ci(nov, BOOT_REPS, SEED)
            row[f"pe_w{w}"] = round(s.mean(), 5)
            row[f"ci_w{w}"] = [round(lo, 5), round(hi, 5)]
        if t in rankings.columns:
            row["percentil_promedio"] = round(float(rankings[t].mean()), 3)
        res[t] = row
    return res


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("Computando PE rolling (3 ventanas x 12 tickers)...")
    frames = compute_series()
    pe140, pe252, pe504 = frames[140], frames[252], frames[504]

    print("\nRankings anuales y rotacion...")
    pct = e6lib.annual_rankings(pe252, list(SECTORS), RANK_YEARS,
                                OUT / "rankings_anuales.csv")
    rot = e6lib.analyze_rotation(pct, SECTORS)
    print(f"  Spearman consecutivo promedio: {rot['spearman_consecutivo_promedio']}")
    print(f"  Permanencia ultimo puesto: {rot['permanencia_ultimo_puesto']} (azar {rot['permanencia_azar']})")

    print("\nPercentiles con CI (las 3 ventanas, robustez)...")
    percs = {}
    for w, pe_w in [(140, pe140), (252, pe252), (504, pe504)]:
        pct_w = pct if w == 252 else e6lib.annual_rankings(
            pe_w, list(SECTORS), RANK_YEARS, OUT / f"rankings_anuales_w{w}.csv")
        percs[f"w{w}"] = e6lib.analyze_percentiles(pct_w, SECTORS, SEED)
    with open(OUT / "percentiles.json", "w") as f:
        json.dump(percs, f, indent=2, ensure_ascii=False)
    for t, r in percs["w252"].items():
        print(f"  {r['sector']:22} {r['percentil_promedio']:.3f} "
              f"CI={r['ci95']} conf={r['confianza_direccional']:.1%} -> {r['senal']}")

    print("\nNiveles con CI...")
    lev = analyze_levels(pe252, pe504, pct)
    for t, r in lev.items():
        if "pe_w252" in r:
            print(f"  {r['sector']:22} PE={r['pe_w252']:.4f} CI={r['ci_w252']}")

    print("\nCrisis...")
    cr = e6lib.analyze_crises(pe140, CRISES, ALL_NAMES, DATA,
                              OUT / "crisis.csv", OUT / "crisis_stats.json",
                              reference=REFERENCE)
    for name in CRISES:
        sub = cr[(cr["episodio"] == name) & (cr["ticker"] != REFERENCE)]
        if len(sub):
            worst = sub.loc[sub["caida"].idxmax()]
            print(f"  {name}: mayor caida {worst['sector']} ({worst['caida']:+.4f})")

    print("\nCo-evolucion (correlacion de cambios mensuales de PE)...")
    corr = e6lib.coevolution(pe252, list(SECTORS), OUT / "coevolucion.csv")
    tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    print(f"  correlacion media entre sectores: {tri.mean():.3f} (rango {tri.min():.3f} a {tri.max():.3f})")

    with open(OUT / "niveles.json", "w") as f:
        json.dump(lev, f, indent=2, ensure_ascii=False)
    with open(OUT / "rotacion.json", "w") as f:
        json.dump(rot, f, indent=2, ensure_ascii=False)
    with open(OUT / "crisis.json", "w") as f:
        json.dump(cr.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    print(f"\nListo -> {OUT}")


if __name__ == "__main__":
    main()
