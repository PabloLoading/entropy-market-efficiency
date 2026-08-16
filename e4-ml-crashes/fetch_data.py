"""E4: descarga de datos para el experimento PE como feature de ML.

1. Baja el snapshot de constituyentes actuales del S&P 500 (Wikipedia) y lo
   guarda en data/sp500_tickers.csv (con fecha de snapshot).
2. Baja daily OHLC ajustado de cada ticker 2000-2025 vía yfinance y guarda
   un parquet por ticker en data/prices/.

Re-ejecutable: saltea tickers ya descargados.
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import io

import pandas as pd
import requests
import yfinance as yf

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PRICES = DATA / "prices"
PRICES.mkdir(parents=True, exist_ok=True)

TICKERS_CSV = DATA / "sp500_tickers.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

START = "2000-01-01"
END = "2025-12-31"
MAX_RETRIES = 3


def fetch_ticker_list() -> pd.DataFrame:
    if TICKERS_CSV.exists():
        print(f"Snapshot ya existe: {TICKERS_CSV}")
        return pd.read_csv(TICKERS_CSV)
    resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"},
                        timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry",
                     "Date added"]].copy()
    df.columns = ["ticker", "name", "sector", "sub_industry", "date_added"]
    # Yahoo usa '-' en lugar de '.' (BRK.B -> BRK-B)
    df["ticker_yahoo"] = df["ticker"].str.replace(".", "-", regex=False)
    df["snapshot_date"] = date.today().isoformat()
    df.to_csv(TICKERS_CSV, index=False)
    print(f"Snapshot guardado: {len(df)} tickers -> {TICKERS_CSV}")
    return df


def fetch_prices(tickers: list[str]) -> None:
    done, failed = 0, []
    for i, t in enumerate(tickers):
        out = PRICES / f"{t}.parquet"
        if out.exists():
            done += 1
            continue
        ok = False
        for attempt in range(MAX_RETRIES):
            try:
                df = yf.download(t, start=START, end=END, interval="1d",
                                 auto_adjust=False, progress=False,
                                 multi_level_index=False)
                if df is not None and len(df) > 0:
                    df.to_parquet(out)
                    ok = True
                else:
                    print(f"  {t}: sin datos")
                    ok = True  # sin datos no es retryable
                break
            except Exception as e:
                print(f"  {t}: intento {attempt + 1} fallo ({e})")
                time.sleep(2 * (attempt + 1))
        if not ok:
            failed.append(t)
        else:
            done += 1
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(tickers)}] procesados")
    print(f"\nListo: {done} ok, {len(failed)} fallidos")
    if failed:
        print("Fallidos:", ", ".join(failed))


def main():
    df = fetch_ticker_list()
    tickers = df["ticker_yahoo"].tolist()
    print(f"Descargando {len(tickers)} tickers ({START} a {END})...")
    fetch_prices(tickers)


if __name__ == "__main__":
    main()
