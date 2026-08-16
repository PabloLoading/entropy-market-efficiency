"""E7: cleanup del dataset SPY 1-min.

Pipeline: dedup exacto -> MT->ET (+2h) -> sesion regular 9:30-15:59 ET ->
excluir medios dias -> excluir dias incompletos (<90% de 390 barras) ->
sanity checks -> parquet 1-min y 5-min. Reporta el conteo de cada paso.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "datasets" / "spy_1min_2008_2021_cleaned.csv"
OUT_1M = HERE / "data" / "spy_1min_regular.parquet"
OUT_5M = HERE / "data" / "spy_5min_regular.parquet"

MIN_BARS = 351  # 90% de 390


def main() -> None:
    df = pd.read_csv(RAW, parse_dates=["date"])
    print(f"crudo: {len(df):,} filas, {df['date'].dt.date.nunique():,} dias "
          f"({df['date'].min()} -> {df['date'].max()})")

    # 1) dedup exacto por timestamp (copias identicas verificadas en el EDA)
    n0 = len(df)
    df = df.drop_duplicates(subset="date", keep="first").sort_values("date")
    print(f"dedup: -{n0 - len(df):,} filas duplicadas -> {len(df):,}")

    # 2) MT -> ET: corrimiento fijo +2h (ambas zonas cambian DST en las mismas fechas)
    df["date"] = df["date"] + pd.Timedelta(hours=2)

    # 3) sesion regular 9:30-15:59 ET (descarta pre-market y after-hours, out of scope)
    t = df["date"].dt.time
    n0 = len(df)
    df = df[(t >= dt.time(9, 30)) & (t < dt.time(16, 0))].copy()
    print(f"sesion regular: -{n0 - len(df):,} barras de horario extendido -> {len(df):,}")

    # 4) medios dias: ultima barra del dia antes de las 15:00 ET
    df["day"] = df["date"].dt.date
    last_bar = df.groupby("day")["date"].max().dt.time
    half_days = set(last_bar[last_bar < dt.time(15, 0)].index)
    df = df[~df["day"].isin(half_days)]
    print(f"medios dias excluidos: {len(half_days)}")

    # 5) dias incompletos (<90% de las 390 barras)
    bpd = df.groupby("day").size()
    bad_days = set(bpd[bpd < MIN_BARS].index)
    df = df[~df["day"].isin(bad_days)]
    print(f"dias incompletos (<{MIN_BARS} barras) excluidos: {len(bad_days)}")

    # 6) sanity checks
    assert df["date"].is_monotonic_increasing, "timestamps no monotonicos"
    assert (df["close"] > 0).all(), "precios no positivos"
    assert not df["date"].dt.dayofweek.isin([5, 6]).any(), "barras de fin de semana"
    dup = df["date"].duplicated().sum()
    assert dup == 0, f"{dup} duplicados restantes"

    bpd = df.groupby("day").size()
    print(f"\npanel limpio: {len(df):,} barras, {df['day'].nunique():,} dias, "
          f"barras/dia mediana={bpd.median():.0f} min={bpd.min()} max={bpd.max()}")
    print(f"rango ET: {df['date'].min()} -> {df['date'].max()}")

    out = df[["date", "open", "high", "low", "close", "volume"]].set_index("date")
    out.to_parquet(OUT_1M)

    # 7) barras 5-min por agregacion (etiqueta = inicio del bucket)
    g = out.resample("5min")
    five = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(), "n_1min": g["close"].count(),
    }).dropna(subset=["close"])
    t5 = five.index.time
    five = five[(t5 >= dt.time(9, 30)) & (t5 < dt.time(16, 0))]
    print(f"barras 5-min: {len(five):,} ({five['n_1min'].eq(5).mean():.1%} con las 5 barras internas)")
    five.to_parquet(OUT_5M)
    print(f"-> {OUT_1M.name}, {OUT_5M.name}")


if __name__ == "__main__":
    main()
