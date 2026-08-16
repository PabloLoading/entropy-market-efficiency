"""E4 EDA parte 1: cobertura y calidad de los datos descargados.

Reporta:
- Tickers descargados vs sin datos.
- Cobertura temporal por ticker (inicio, fin, filas).
- Cuántos tienen historia completa desde 2000 vs parciales (IPOs posteriores).
- Calidad: NaNs, precios <= 0, retornos diarios extremos (posibles splits mal ajustados).
Genera data/eda_coverage.csv y charts en outputs/charts/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PRICES = DATA / "prices"
CHARTS = HERE / "outputs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

EXTREME_RET = 0.50  # |retorno diario| > 50% se cuenta como sospechoso


def main():
    tickers_df = pd.read_csv(DATA / "sp500_tickers.csv")
    rows = []
    for _, r in tickers_df.iterrows():
        t = r["ticker_yahoo"]
        f = PRICES / f"{t}.parquet"
        if not f.exists():
            rows.append({"ticker": t, "sector": r["sector"], "status": "missing"})
            continue
        df = pd.read_parquet(f)
        if len(df) == 0:
            rows.append({"ticker": t, "sector": r["sector"], "status": "empty"})
            continue
        close = df["Adj Close"].dropna()
        ret = close.pct_change().dropna()
        rows.append({
            "ticker": t,
            "sector": r["sector"],
            "status": "ok",
            "n_rows": len(df),
            "start": df.index.min().date(),
            "end": df.index.max().date(),
            "nan_close": int(df["Adj Close"].isna().sum()),
            "nonpos_close": int((close <= 0).sum()),
            "extreme_ret_days": int((ret.abs() > EXTREME_RET).sum()),
        })
    cov = pd.DataFrame(rows)
    cov.to_csv(DATA / "eda_coverage.csv", index=False)

    ok = cov[cov["status"] == "ok"].copy()
    print(f"Tickers total: {len(cov)}")
    print(f"  ok:      {len(ok)}")
    print(f"  missing: {(cov['status'] == 'missing').sum()}")
    print(f"  empty:   {(cov['status'] == 'empty').sum()}")

    ok["start"] = pd.to_datetime(ok["start"])
    full = (ok["start"] <= "2000-01-10").sum()
    print(f"\nHistoria completa desde 2000: {full} tickers")
    print(f"Listados despues de 2000 (IPO posterior): {len(ok) - full}")
    print("\nDistribucion de anio de inicio:")
    print(ok["start"].dt.year.value_counts().sort_index().to_string())

    print(f"\nCalidad:")
    print(f"  tickers con NaNs en Adj Close: {(ok['nan_close'] > 0).sum()}")
    print(f"  tickers con precios <= 0:      {(ok['nonpos_close'] > 0).sum()}")
    print(f"  tickers con dias |ret|>50%:    {(ok['extreme_ret_days'] > 0).sum()}")
    worst = ok.nlargest(10, "extreme_ret_days")[["ticker", "extreme_ret_days"]]
    print("\nTop 10 con mas dias extremos:")
    print(worst.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ok["start"].dt.year.value_counts().sort_index().plot.bar(ax=ax, color="steelblue")
    ax.set_xlabel("Año de inicio de la serie")
    ax.set_ylabel("Cantidad de tickers")
    ax.set_title("Cobertura: año de inicio de la historia por ticker")
    plt.tight_layout()
    fig.savefig(CHARTS / "eda_start_years.png", dpi=150)
    plt.close(fig)
    print(f"\nChart guardado: {CHARTS / 'eda_start_years.png'}")


if __name__ == "__main__":
    main()
