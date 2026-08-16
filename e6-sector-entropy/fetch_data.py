"""E6: descarga de los 11 ETFs SPDR sectoriales + SPY (referencia), diario 1998-2025.

Re-ejecutable: saltea tickers ya descargados. Reporta cobertura, en particular
las fechas de inicio de XLRE y XLC (panel desbalanceado, declarado).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

TICKERS = {
    "XLK": "Tecnología", "XLE": "Energía", "XLV": "Salud", "XLF": "Financiero",
    "XLP": "Consumo básico", "XLY": "Consumo discrecional", "XLI": "Industrial",
    "XLU": "Utilities", "XLB": "Materiales", "XLRE": "Real Estate",
    "XLC": "Comunicaciones", "SPY": "S&P 500 (referencia)",
}
# Sub-experimento 2: indices nativos de las 9 bolsas. Clave = nombre
# de archivo seguro; valor = (ticker Yahoo, nombre de mercado).
MARKETS = {
    "us": ("^GSPC", "Estados Unidos"), "japon": ("^N225", "Japón"),
    "uk": ("^FTSE", "Reino Unido"), "hongkong": ("^HSI", "Hong Kong"),
    "alemania": ("^GDAXI", "Alemania"), "australia": ("^AXJO", "Australia"),
    "brasil": ("^BVSP", "Brasil"), "china": ("000001.SS", "China"),
    "india": ("^BSESN", "India"),
}
START = "1998-01-01"
END = "2025-12-31"
MAX_RETRIES = 3


def fetch_markets() -> None:
    mdir = DATA / "markets"
    mdir.mkdir(exist_ok=True)
    print("\nDescargando indices nativos (1998-2025)...")
    for key, (ticker, name) in MARKETS.items():
        out = mdir / f"{key}.parquet"
        if out.exists():
            df = pd.read_parquet(out)
            print(f"  {name:15} ({ticker}): ya existe, {df.index[0].date()} -> {df.index[-1].date()}  n={len(df)}")
            continue
        for attempt in range(MAX_RETRIES):
            try:
                df = yf.download(ticker, start=START, end=END, interval="1d",
                                 auto_adjust=False, progress=False,
                                 multi_level_index=False)
                if df is not None and len(df) > 0:
                    df.to_parquet(out)
                    print(f"  {name:15} ({ticker}): {df.index[0].date()} -> {df.index[-1].date()}  n={len(df)}")
                break
            except Exception as e:
                print(f"  {name}: intento {attempt + 1} fallo ({e})")
                time.sleep(2 * (attempt + 1))


def main() -> None:
    failed = []
    print("Descargando ETFs sectoriales...")
    for t, name in TICKERS.items():
        out = DATA / f"{t}.parquet"
        if out.exists():
            df = pd.read_parquet(out)
            print(f"  {t:5} ({name}): ya existe, {df.index[0].date()} -> {df.index[-1].date()}  n={len(df)}")
            continue
        ok = False
        for attempt in range(MAX_RETRIES):
            try:
                df = yf.download(t, start=START, end=END, interval="1d",
                                 auto_adjust=False, progress=False,
                                 multi_level_index=False)
                if df is not None and len(df) > 0:
                    df.to_parquet(out)
                    print(f"  {t:5} ({name}): {df.index[0].date()} -> {df.index[-1].date()}  n={len(df)}")
                    ok = True
                break
            except Exception as e:
                print(f"  {t}: intento {attempt + 1} fallo ({e})")
                time.sleep(2 * (attempt + 1))
        if not ok and not out.exists():
            failed.append(t)
    if failed:
        print("Fallidos:", ", ".join(failed))
    else:
        print("Listo: 12/12.")
    fetch_markets()


if __name__ == "__main__":
    main()
