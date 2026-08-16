"""E4 tarea 1.1: dataset model-ready (feature store lite).

Toma data/events_raw.csv + data/prices/*.parquet y produce
data/events_features.parquet con una fila por evento:
- 6 features point-in-time (pe_t en W=140 y W=180, drawdown_actual_t,
  vol_realizada_t, reversal_5d_t, market_dd_t, rel_volume_t)
- target trough_final
- metadata: ticker, sector, fechas, episodio sistemico, peso 1/k

Toda feature usa solo datos con fecha <= entry_date del evento.
PE via common.perm_entropy (sin reimplementar).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# import directo del submodulo: common/__init__ arrastra lingam (no instalado aca)
sys.path.insert(0, str(HERE.parent / "common"))

from entropy import perm_entropy  # noqa: E402

DATA = HERE / "data"
PRICES = DATA / "prices"

PE_WINDOWS = [140, 180]
PE_M = 3
VOL_WINDOW_5D = 5
RELVOL_SHORT = 5
RELVOL_LONG = 60
MARKET_EPISODE_DD = 0.10
MIN_RET_PEAK_ENTRY = 5


def market_dd() -> pd.Series:
    close = pd.read_parquet(DATA / "market_gspc.parquet")["Adj Close"].dropna()
    return close / close.rolling(252, min_periods=1).max() - 1


def build_episodes(mdd: pd.Series) -> pd.Series:
    """Mapa fecha -> episode_id para dias con mdd <= -10% (runs contiguos)."""
    in_ep = mdd <= -MARKET_EPISODE_DD
    ep_id = (in_ep != in_ep.shift()).cumsum()
    ep = pd.Series(np.where(in_ep, "SYS_" + ep_id.astype(str), None),
                   index=mdd.index)
    return ep


def main():
    events = pd.read_csv(DATA / "events_raw.csv",
                         parse_dates=["peak_date", "entry_date", "trough_date"])
    mdd = market_dd()
    episodes = build_episodes(mdd)

    rows = []
    for ticker, ev_t in events.groupby("ticker"):
        f = PRICES / f"{ticker}.parquet"
        if not f.exists():
            continue
        px = pd.read_parquet(f)
        close = px["Adj Close"].dropna()
        volume = px["Volume"].reindex(close.index)
        log_ret = np.log(close / close.shift(1)).dropna()

        for _, ev in ev_t.iterrows():
            entry = ev["entry_date"]
            peak = ev["peak_date"]
            if entry not in close.index or peak not in close.index:
                continue

            ret_hist = log_ret.loc[:entry]
            row = {
                "ticker": ticker,
                "sector": ev["sector"],
                "peak_date": peak,
                "entry_date": entry,
                "trough_date": ev["trough_date"],
                "year": ev["year"],
                "duration_days": ev["duration_days"],
                "trough_final": ev["trough_final"],
                "is_crash": ev["is_crash"],
            }

            for w in PE_WINDOWS:
                win = ret_hist.iloc[-w:]
                if len(win) == w:
                    row[f"pe_t_w{w}"] = perm_entropy(win.values, m=PE_M, tau=1)
                    row[f"wpe_t_w{w}"] = perm_entropy(win.values, m=PE_M, tau=1,
                                                       weighted=True)
                else:
                    row[f"pe_t_w{w}"] = np.nan
                    row[f"wpe_t_w{w}"] = np.nan

            row["drawdown_actual_t"] = float(close.loc[entry] / close.loc[peak] - 1)

            # vol del evento en curso; si el evento es mas corto que 5 ruedas
            # se completa con ventana trailing hasta la entrada (point-in-time)
            ret_peak_entry = log_ret.loc[peak:entry]
            if len(ret_peak_entry) < MIN_RET_PEAK_ENTRY:
                ret_peak_entry = ret_hist.iloc[-MIN_RET_PEAK_ENTRY:]
            row["vol_realizada_t"] = (float(ret_peak_entry.std())
                                       if len(ret_peak_entry) >= MIN_RET_PEAK_ENTRY
                                       else np.nan)

            row["reversal_5d_t"] = (float(ret_hist.iloc[-VOL_WINDOW_5D:].sum())
                                     if len(ret_hist) >= VOL_WINDOW_5D else np.nan)

            mdd_entry = mdd.reindex([entry])
            row["market_dd_t"] = float(mdd_entry.iloc[0]) if np.isfinite(
                mdd_entry.iloc[0]) else np.nan

            vol_hist = volume.loc[:entry].dropna()
            if len(vol_hist) >= RELVOL_LONG and vol_hist.iloc[-RELVOL_LONG:].mean() > 0:
                row["rel_volume_t"] = float(
                    vol_hist.iloc[-RELVOL_SHORT:].mean()
                    / vol_hist.iloc[-RELVOL_LONG:].mean())
            else:
                row["rel_volume_t"] = np.nan

            ep = episodes.reindex([entry]).iloc[0]
            is_sys = isinstance(ep, str)  # None/NaN = fuera de episodio
            row["episode_id"] = ep if is_sys else f"IDIO_{ticker}_{entry.date()}"
            row["systemic"] = is_sys
            rows.append(row)

    df = pd.DataFrame(rows)
    k = df.groupby("episode_id")["ticker"].transform("size")
    df["weight"] = 1.0 / k

    out = DATA / "events_features.parquet"
    df.to_parquet(out, index=False)

    print(f"Eventos de entrada:  {len(events)}")
    print(f"Filas generadas:     {len(df)}")
    feat_cols = ([f"pe_t_w{w}" for w in PE_WINDOWS]
                 + [f"wpe_t_w{w}" for w in PE_WINDOWS]
                 + ["drawdown_actual_t", "vol_realizada_t", "reversal_5d_t",
                    "market_dd_t", "rel_volume_t"])
    print("\nNaNs por feature:")
    for c in feat_cols:
        print(f"  {c:22s} {df[c].isna().sum():>6d} ({df[c].isna().mean():.1%})")
    for w in PE_WINDOWS:
        complete = df.dropna(subset=[f"pe_t_w{w}", "drawdown_actual_t",
                                      "vol_realizada_t", "reversal_5d_t",
                                      "market_dd_t", "rel_volume_t"])
        print(f"\nFilas completas W={w}: {len(complete)}")
    print(f"\nEpisodios sistemicos: {df[df['systemic']]['episode_id'].nunique()}")
    print(f"Eventos sistemicos:   {df['systemic'].sum()} ({df['systemic'].mean():.1%})")
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
