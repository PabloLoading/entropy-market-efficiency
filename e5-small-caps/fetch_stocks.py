"""E5 fase 2: descarga de las acciones para el estudio multi-asset.

1. Snapshot de constituyentes actuales del S&P 600 (Wikipedia) -> data/sp600_tickers.csv.
2. Daily OHLCV de cada ticker 2015-2025 -> un parquet por ticker en data/prices_sp600/.
3. Chequeo de solapamiento con la lista del S&P 500 de E4.

Los precios del S&P 500 se reusan de ../e4-ml-crashes/data/prices/ (2000-2025 incluye
2015-2025). Re-ejecutable: saltea tickers ya descargados.
"""
from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PRICES_600 = DATA / "prices_sp600"
PRICES_600.mkdir(parents=True, exist_ok=True)

E4_DATA = HERE.parent / "e4-ml-crashes" / "data"
SP500_TICKERS_CSV = E4_DATA / "sp500_tickers.csv"

TICKERS_CSV = DATA / "sp600_tickers.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"

START = "2015-01-01"
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
    # La tabla de constituyentes es la que tiene la columna Symbol.
    df = None
    for t in tables:
        if "Symbol" in t.columns:
            df = t
            break
    if df is None:
        raise SystemExit("ERROR: no se encontro la tabla de constituyentes en Wikipedia.")
    cols = {"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"}
    df = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    df["ticker_yahoo"] = df["ticker"].str.replace(".", "-", regex=False)
    df["snapshot_date"] = date.today().isoformat()
    df.to_csv(TICKERS_CSV, index=False)
    print(f"Snapshot guardado: {len(df)} tickers -> {TICKERS_CSV}")
    return df


def check_overlap(sp600: pd.DataFrame) -> None:
    if not SP500_TICKERS_CSV.exists():
        print("AVISO: no se encontro la lista del S&P 500 de E4; sin chequeo de solapamiento.")
        return
    sp500 = pd.read_csv(SP500_TICKERS_CSV)
    overlap = set(sp600["ticker_yahoo"]) & set(sp500["ticker_yahoo"])
    if overlap:
        print(f"AVISO: {len(overlap)} tickers en ambas listas (se excluiran del "
              f"grupo small en el analisis): {sorted(overlap)}")
    else:
        print("Solapamiento S&P 600 vs S&P 500: ninguno.")


def fetch_prices(tickers: list[str]) -> None:
    done, failed, empty = 0, [], []
    for i, t in enumerate(tickers):
        out = PRICES_600 / f"{t}.parquet"
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
                else:
                    empty.append(t)
                ok = True
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
    print(f"\nListo: {done} ok ({len(empty)} sin datos), {len(failed)} fallidos")
    if empty:
        print("Sin datos:", ", ".join(empty))
    if failed:
        print("Fallidos:", ", ".join(failed))


def main() -> None:
    df = fetch_ticker_list()
    check_overlap(df)
    tickers = df["ticker_yahoo"].tolist()
    print(f"Descargando {len(tickers)} tickers ({START} a {END})...")
    fetch_prices(tickers)


if __name__ == "__main__":
    main()
