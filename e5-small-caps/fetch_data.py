"""Fetch daily close series for the S&P 600 (^SML) and S&P 500 (^GSPC) indices.

Re-runnable: overwrites the parquets in data/. Prints a coverage summary and
checks for pre-launch backfill and large gaps (spec e5-index-data).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# S&P SmallCap 600 launched 1994-10-28; anything earlier would be backfill.
SP600_LAUNCH = pd.Timestamp("1994-10-28")
END = "2025-12-31"

INDICES = {
    "sp600": "^SP600",  # S&P 600 (small caps); ^SML no existe en Yahoo
    "sp500": "^GSPC",   # S&P 500 (large caps)
}
# Fallback ETFs if ^SML is missing or too thin (spec: start after 2005 or
# gaps > 10 business days).
FALLBACK = {"sp600": "IJR", "sp500": "SPY"}


def fetch_close(ticker: str) -> pd.Series | None:
    df = yf.download(ticker, start="1900-01-01", end=END, interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.rename("close")


def coverage_report(name: str, s: pd.Series) -> dict:
    gaps = s.index.to_series().diff().dt.days.dropna()
    max_gap = int(gaps.max()) if len(gaps) else 0
    big_gaps = gaps[gaps > 14]  # >14 calendar days ~ >10 business days
    info = {
        "start": s.index[0],
        "end": s.index[-1],
        "n": len(s),
        "max_gap_days": max_gap,
        "n_big_gaps": len(big_gaps),
    }
    print(f"  {name}: {info['start'].date()} -> {info['end'].date()}  "
          f"n={info['n']}  max_gap={max_gap}d  gaps>10bd={len(big_gaps)}")
    for dt, g in big_gaps.items():
        print(f"    gap de {int(g)} dias calendario terminando en {dt.date()}")
    return info


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print("Descargando indices...")

    for name, ticker in INDICES.items():
        s = fetch_close(ticker)
        used = ticker

        if name == "sp600" and s is not None:
            # Truncate any pre-launch backfill.
            pre = s[s.index < SP600_LAUNCH]
            if len(pre):
                print(f"  AVISO: {ticker} tiene {len(pre)} obs previas al "
                      f"lanzamiento del S&P 600 ({SP600_LAUNCH.date()}); se truncan.")
                s = s[s.index >= SP600_LAUNCH]

        insufficient = (
            s is None
            or s.index[0] > pd.Timestamp("2005-01-01")
            or (s.index.to_series().diff().dt.days.dropna() > 14).sum() > 0
        )
        if insufficient:
            fb = FALLBACK[name]
            print(f"  AVISO: {ticker} insuficiente; fallback a {fb} "
                  f"(revisar la fuente si esto ocurre).")
            s = fetch_close(fb)
            used = fb
            if s is None:
                raise SystemExit(f"ERROR: sin datos para {name} ({ticker} ni {fb}).")

        info = coverage_report(f"{name} ({used})", s)
        out = DATA_DIR / f"{name}.parquet"
        s.to_frame().to_parquet(out)
        print(f"    -> {out.name}  (fuente: {used})")

    print("\nListo.")


if __name__ == "__main__":
    main()
