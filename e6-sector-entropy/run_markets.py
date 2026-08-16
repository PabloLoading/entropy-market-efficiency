"""E6 sub-experimento 2: PE de los 9 indices nativos, con la
maquinaria compartida de e6lib.

Produce en outputs/:
- pe_markets_w{140,252,504}.csv    series rolling de PE por mercado
- rankings_mercados*.csv           percentil de cada mercado por año (2000-2025)
- markets_percentiles.json         percentil promedio con CI + tendencia Mann-Kendall
- markets_rotacion.json            rotacion del ranking de mercados
- markets_crisis.csv/_stats.json   caidas de PE vs drawdown (4 globales + China 2015)
- markets_flujo.json               co-movimiento contemporaneo y asimetria lag-1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import e6lib
from fetch_data import MARKETS  # {clave: (ticker, nombre)}

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "markets"
OUT = HERE / "outputs"

NAMES = {k: name for k, (_, name) in MARKETS.items()}
WINDOWS = [140, 252, 504]
RANK_YEARS = range(2000, 2026)
SEED = 42

CRISES = {
    "dot-com": ("2000-03-24", "2002-10-09"),
    "crisis-2008": ("2007-10-09", "2009-03-09"),
    "covid": ("2020-02-19", "2020-03-23"),
    "tasas-2022": ("2022-01-03", "2022-10-12"),
    "china-2015 (local)": ("2015-06-12", "2016-01-28"),
}


def compute_series() -> dict[int, pd.DataFrame]:
    if all((OUT / f"pe_markets_w{w}.csv").exists() for w in WINDOWS):
        print("  (series cacheadas, salteando computo)")
        return {w: pd.read_csv(OUT / f"pe_markets_w{w}.csv", index_col="date",
                               parse_dates=True) for w in WINDOWS}
    series: dict[int, dict[str, pd.Series]] = {w: {} for w in WINDOWS}
    rets = {}
    for k in MARKETS:
        close = pd.read_parquet(DATA / f"{k}.parquet")["Adj Close"].dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        ret = np.log(close / close.shift(1)).dropna()
        rets[k] = ret
        for w in WINDOWS:
            series[w][k] = e6lib.rolling_pe(ret, w)
        print(f"  {NAMES[k]}: PE rolling listo ({len(ret)} retornos)")
    frames = {}
    for w in WINDOWS:
        df = pd.DataFrame(series[w]).dropna(how="all")
        df.to_csv(OUT / f"pe_markets_w{w}.csv", index_label="date")
        frames[w] = df
    return frames


def flow_analysis(pe252: pd.DataFrame) -> dict:
    """Sondeo descriptivo de flujo: co-movimiento contemporaneo de cambios
    mensuales de PE (contra retornos) y asimetria lag-1 con US como emisor."""
    monthly_pe = pe252[list(MARKETS)].resample("ME").last()
    dpe = monthly_pe.diff().dropna(how="all")
    # retornos mensuales para el contraste
    rets = {}
    for k in MARKETS:
        px = pd.read_parquet(DATA / f"{k}.parquet")["Adj Close"].dropna()
        px.index = pd.to_datetime(px.index).tz_localize(None)
        rets[k] = np.log(px / px.shift(1)).resample("ME").sum()
    dret = pd.DataFrame(rets)

    def mean_offdiag(df):
        c = df.corr(min_periods=24)
        tri = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack().dropna()
        return round(float(tri.mean()), 3), round(float(tri.min()), 3), round(float(tri.max()), 3)

    pe_m, pe_lo, pe_hi = mean_offdiag(dpe)
    r_m, r_lo, r_hi = mean_offdiag(dret)

    # asimetria lag-1: US(t) vs otro(t+1), y otro(t) vs US(t+1)
    us_lead, us_lag = [], []
    for k in MARKETS:
        if k == "us":
            continue
        a = dpe["us"].corr(dpe[k].shift(-1))   # US hoy -> otro el mes siguiente
        b = dpe[k].corr(dpe["us"].shift(-1))   # otro hoy -> US el mes siguiente
        if not np.isnan(a):
            us_lead.append(float(a))
        if not np.isnan(b):
            us_lag.append(float(b))
    res = {
        "corr_media_dpe": {"media": pe_m, "rango": [pe_lo, pe_hi]},
        "corr_media_retornos": {"media": r_m, "rango": [r_lo, r_hi]},
        "asimetria_lag1": {
            "us_a_otros": round(float(np.mean(us_lead)), 3),
            "otros_a_us": round(float(np.mean(us_lag)), 3),
            "detalle_us_a_otros": {NAMES[k]: round(float(dpe['us'].corr(dpe[k].shift(-1))), 3)
                                    for k in MARKETS if k != "us"},
        },
    }
    with open(OUT / "markets_flujo.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return res


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("Computando PE rolling (3 ventanas x 9 mercados)...")
    frames = compute_series()
    pe140, pe252, pe504 = frames[140], frames[252], frames[504]

    print("\nRankings, percentiles y tendencia...")
    pct = e6lib.annual_rankings(pe252, list(MARKETS), RANK_YEARS,
                                OUT / "rankings_mercados.csv")
    pct140 = e6lib.annual_rankings(pe140, list(MARKETS), RANK_YEARS,
                                   OUT / "rankings_mercados_w140.csv")
    pct504 = e6lib.annual_rankings(pe504, list(MARKETS), RANK_YEARS,
                                   OUT / "rankings_mercados_w504.csv")
    percs = {"w140": e6lib.analyze_percentiles(pct140, NAMES, SEED),
             "w252": e6lib.analyze_percentiles(pct, NAMES, SEED),
             "w504": e6lib.analyze_percentiles(pct504, NAMES, SEED)}
    # tendencia Mann-Kendall sobre la PE media anual (W=252) de cada mercado
    annual = pe252[list(MARKETS)].groupby(pe252.index.year).mean()
    annual = annual.loc[(annual.index >= 2000) & (annual.index <= 2025)]
    for k in MARKETS:
        percs["w252"][k]["mann_kendall"] = e6lib.mann_kendall(annual[k])
    # robust check estilo Zunino: PE estatica m=6 sobre la serie completa 1998-2025
    from e6lib import perm_entropy
    for k in MARKETS:
        px = pd.read_parquet(DATA / f"{k}.parquet")["Adj Close"].dropna()
        px.index = pd.to_datetime(px.index).tz_localize(None)
        ret = np.log(px / px.shift(1)).dropna()
        percs["w252"][k]["pe_estatica_m6"] = round(
            float(perm_entropy(ret.values, m=6, tau=1, normalize=True)), 4)
    with open(OUT / "markets_percentiles.json", "w") as f:
        json.dump(percs, f, indent=2, ensure_ascii=False)
    for k, r in sorted(percs["w252"].items(), key=lambda kv: -kv[1]["percentil_promedio"]):
        mk = r["mann_kendall"]
        print(f"  {r['sector']:15} {r['percentil_promedio']:.3f} CI={r['ci95']} "
              f"conf={r['confianza_direccional']:.1%} | MK: {mk['tendencia']} (p={mk['p']})")

    rot = e6lib.analyze_rotation(pct, NAMES)
    with open(OUT / "markets_rotacion.json", "w") as f:
        json.dump(rot, f, indent=2, ensure_ascii=False)
    print(f"  rotacion: spearman={rot['spearman_consecutivo_promedio']} "
          f"permanencia={rot['permanencia_ultimo_puesto']} (azar {rot['permanencia_azar']})")

    print("\nCrisis (4 globales + China 2015 local)...")
    cr = e6lib.analyze_crises(pe140, CRISES, NAMES, DATA,
                              OUT / "markets_crisis.csv",
                              OUT / "markets_crisis_stats.json")
    for name in CRISES:
        sub = cr[cr["episodio"] == name]
        if len(sub):
            worst = sub.loc[sub["caida"].idxmax()]
            print(f"  {name}: mayor caida {worst['sector']} ({worst['caida']:+.4f})")

    print("\nSondeo de flujo...")
    flow = flow_analysis(pe252)
    print(f"  corr media dPE: {flow['corr_media_dpe']['media']} | "
          f"retornos: {flow['corr_media_retornos']['media']}")
    print(f"  lag-1 US->otros: {flow['asimetria_lag1']['us_a_otros']} | "
          f"otros->US: {flow['asimetria_lag1']['otros_a_us']}")
    print(f"\nListo -> {OUT}")


if __name__ == "__main__":
    main()
