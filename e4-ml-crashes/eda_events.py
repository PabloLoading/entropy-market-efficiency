"""E4 EDA parte 2: deteccion de eventos y estructura temporal.

Responde:
- Cuantas observaciones (eventos) tenemos en total y tras el filtro de duracion.
- Como se distribuyen por anio y por stock.
- Que fraccion cae dentro de episodios sistemicos (S&P 500 en dd >= 10%).
- Con que splits walk-forward quedaria el n de train/val/test por fold.

Genera data/events_raw.csv y charts en outputs/charts/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PRICES = DATA / "prices"
CHARTS = HERE / "outputs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

MIN_DD = 0.05
MAX_DAYS = 252
MIN_EVENT_DAYS = 15
MARKET_DD_EPISODE = 0.10
CRASH_CUTOFF = 0.15
START, END = "2000-01-01", "2025-12-31"


def detect_events(close: pd.Series, min_dd=MIN_DD, max_days=MAX_DAYS) -> pd.DataFrame:
    """Peak tracker con reset post-evento (igual a entropy-crashes)."""
    dates = close.index
    p = close.values
    n = len(p)
    events = []
    peak_price = p[0]
    peak_idx = 0
    i = 1
    while i < n:
        if p[i] > peak_price:
            peak_price = p[i]
            peak_idx = i
            i += 1
            continue
        if p[i] / peak_price - 1 <= -min_dd:
            entry_idx = i
            trough_idx = i
            trough_price = p[i]
            j = i + 1
            while j < n:
                if p[j] < trough_price:
                    trough_price = p[j]
                    trough_idx = j
                if p[j] >= peak_price:
                    break
                if j - peak_idx >= max_days:
                    break
                j += 1
            recovery_idx = j if j < n else n - 1
            events.append({
                "peak_date": dates[peak_idx],
                "entry_date": dates[entry_idx],
                "trough_date": dates[trough_idx],
                "drawdown": (trough_price - peak_price) / peak_price,
                "duration_days": int((dates[trough_idx] - dates[peak_idx]).days),
            })
            i = recovery_idx + 1
            if i < n:
                peak_price = p[recovery_idx]
                peak_idx = recovery_idx
            continue
        i += 1
    return pd.DataFrame(events)


def market_dd_series() -> pd.Series:
    f = DATA / "market_gspc.parquet"
    if f.exists():
        close = pd.read_parquet(f)["Adj Close"]
    else:
        df = yf.download("^GSPC", start=START, end=END, interval="1d",
                         auto_adjust=False, progress=False,
                         multi_level_index=False)
        df.to_parquet(f)
        close = df["Adj Close"]
    close = close.dropna()
    # dd contra maximo rolling de 252d: captura estres reciente, no anios
    # underwater tras un crash grande (ej. 2003-2006 post dot-com)
    return close / close.rolling(252, min_periods=1).max() - 1


def main():
    tickers_df = pd.read_csv(DATA / "sp500_tickers.csv")
    mdd = market_dd_series()

    all_events = []
    for _, r in tickers_df.iterrows():
        t = r["ticker_yahoo"]
        f = PRICES / f"{t}.parquet"
        if not f.exists():
            continue
        close = pd.read_parquet(f)["Adj Close"].dropna()
        if len(close) < 300:
            continue
        ev = detect_events(close)
        if len(ev) == 0:
            continue
        ev["ticker"] = t
        ev["sector"] = r["sector"]
        all_events.append(ev)

    events = pd.concat(all_events, ignore_index=True)
    n_raw = len(events)
    events = events[events["duration_days"] >= MIN_EVENT_DAYS].copy()
    n_filt = len(events)

    # episodio sistemico: dd del S&P 500 en la fecha de entrada
    events["market_dd_entry"] = mdd.reindex(events["entry_date"]).values
    events["systemic"] = events["market_dd_entry"] <= -MARKET_DD_EPISODE
    events["year"] = events["entry_date"].dt.year
    events["trough_final"] = events["drawdown"].abs()
    events["is_crash"] = events["trough_final"] >= CRASH_CUTOFF
    events.to_csv(DATA / "events_raw.csv", index=False)

    print(f"Eventos brutos (dd>={MIN_DD:.0%}):            {n_raw}")
    print(f"Tras filtro duracion >={MIN_EVENT_DAYS}d:      {n_filt}")
    print(f"\nEventos por stock: mediana {events.groupby('ticker').size().median():.0f}, "
          f"media {events.groupby('ticker').size().mean():.1f}, "
          f"max {events.groupby('ticker').size().max()}")
    print(f"\nSistemicos (S&P500 dd>={MARKET_DD_EPISODE:.0%} en entrada): "
          f"{events['systemic'].sum()} ({events['systemic'].mean():.1%})")
    print(f"Idiosincraticos: {(~events['systemic']).sum()} "
          f"({(~events['systemic']).mean():.1%})")
    print(f"\nDistribucion trough_final: mediana {events['trough_final'].median():.3f}, "
          f"p90 {events['trough_final'].quantile(0.9):.3f}, "
          f"max {events['trough_final'].max():.3f}")
    print(f"Crashes (>={CRASH_CUTOFF:.0%}): {events['is_crash'].sum()} "
          f"({events['is_crash'].mean():.1%})")

    print("\nEventos por anio (total / sistemicos):")
    by_year = events.groupby("year").agg(total=("ticker", "size"),
                                          systemic=("systemic", "sum"))
    print(by_year.to_string())

    # simulacion de splits walk-forward: test en bloques de 2 anios desde 2009
    print("\nWalk-forward tentativo (train expandiente desde 2000, "
          "val = ultimo 20% del train, embargo 1 anio, test 2 anios):")
    print(f"{'fold':>4s} {'train':>18s} {'n_tr':>6s} {'test':>12s} {'n_te':>6s}")
    fold = 1
    for test_start in range(2009, 2025, 2):
        train_end = test_start - 1  # embargo de 1 anio
        n_tr = (events["year"] <= train_end - 1).sum()
        n_te = events["year"].isin([test_start, test_start + 1]).sum()
        print(f"{fold:>4d} 2000-{train_end - 1:<12d} {n_tr:>6d} "
              f"{test_start}-{test_start + 1:>4d} {n_te:>6d}")
        fold += 1

    # charts
    fig, ax = plt.subplots(figsize=(9, 4.8))
    by_year["total"].plot.bar(ax=ax, color="grey", label="idiosincraticos")
    by_year["systemic"].plot.bar(ax=ax, color="steelblue", label="sistemicos")
    ax.set_xlabel("Año de entrada del evento")
    ax.set_ylabel("Eventos")
    ax.set_title("Eventos por año (sistemico = S&P 500 en dd ≥ 10% en la entrada)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(CHARTS / "eda_events_per_year.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(events["trough_final"], bins=60, color="steelblue", edgecolor="black",
             linewidth=0.3)
    ax.axvline(CRASH_CUTOFF, color="red", ls="--", lw=1,
                label=f"crash cutoff {CRASH_CUTOFF:.0%}")
    ax.set_xlabel("Trough final (|drawdown|)")
    ax.set_ylabel("Eventos")
    ax.set_title("Distribución de la profundidad final del evento")
    ax.legend()
    plt.tight_layout()
    fig.savefig(CHARTS / "eda_trough_dist.png", dpi=150)
    plt.close(fig)
    print(f"\nCharts guardados en {CHARTS}")


if __name__ == "__main__":
    main()
