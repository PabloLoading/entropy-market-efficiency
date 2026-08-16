"""E5: rolling PE of the S&P 600 vs S&P 500 (descriptivo).

Outputs to outputs/:
- pe_series_w{W}.csv   rolling PE of both indices + diff (per config)
- annual_gap_w{W}.csv  yearly mean/std of the diff
- results.json         paired diff on non-overlapping windows, CI 95% (block bootstrap)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from entropy import perm_entropy  # noqa: E402

DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"

M = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # robustez D12: correr con 4
TAU = 1
WINDOWS = [252, 504]
BOOT_REPS = 2000
SEED = 42
SUFFIX = "" if M == 3 else f"_m{M}"


def load_aligned() -> pd.DataFrame:
    sp600 = pd.read_parquet(DATA_DIR / "sp600.parquet")["close"]
    sp500 = pd.read_parquet(DATA_DIR / "sp500.parquet")["close"]
    df = pd.concat({"sp600": sp600, "sp500": sp500}, axis=1)
    n600, n500 = df["sp600"].notna().sum(), df["sp500"].notna().sum()
    df = df.dropna()
    print(f"Alineacion: {len(df)} fechas comunes "
          f"(descartadas: sp600 {n600 - len(df)}, sp500 {n500 - len(df)})")
    return df


def rolling_pe(returns: pd.Series, w: int) -> pd.Series:
    vals = returns.values
    out = np.full(len(vals), np.nan)
    for i in range(w, len(vals) + 1):
        out[i - 1] = perm_entropy(vals[i - w:i], m=M, tau=TAU, normalize=True)
    return pd.Series(out, index=returns.index, name="pe")


def block_bootstrap_ci(diff: pd.Series, reps: int, seed: int) -> tuple[float, float]:
    """CI 95% of the mean paired diff, resampling calendar years with replacement."""
    years = diff.index.year
    blocks = [diff.values[years == y] for y in np.unique(years)]
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    k = len(blocks)
    for r in range(reps):
        idx = rng.integers(0, k, size=k)
        sample = np.concatenate([blocks[i] for i in idx])
        means[r] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    prices = load_aligned()
    returns = np.log(prices / prices.shift(1)).dropna()

    results = {}
    for w in WINDOWS:
        print(f"\n--- W={w} ---")
        pe600 = rolling_pe(returns["sp600"], w)
        pe500 = rolling_pe(returns["sp500"], w)
        df = pd.concat({"pe_sp600": pe600, "pe_sp500": pe500}, axis=1).dropna()
        df["diff"] = df["pe_sp600"] - df["pe_sp500"]
        df.to_csv(OUT_DIR / f"pe_series_w{w}{SUFFIX}.csv", index_label="date")

        # Non-overlapping windows: one observation every w days, paired by date.
        nov = df.iloc[::w]
        ci_lo, ci_hi = block_bootstrap_ci(nov["diff"], BOOT_REPS, SEED)

        annual = df["diff"].groupby(df.index.year).agg(["mean", "std", "count"])
        annual.to_csv(OUT_DIR / f"annual_gap_w{w}{SUFFIX}.csv", index_label="year")

        results[f"w{w}"] = {
            "pe_sp600_mean": round(df["pe_sp600"].mean(), 6),
            "pe_sp500_mean": round(df["pe_sp500"].mean(), 6),
            "diff_mean_all": round(df["diff"].mean(), 6),
            "diff_mean_nonoverlap": round(nov["diff"].mean(), 6),
            "n_nonoverlap": len(nov),
            "ci95": [round(ci_lo, 6), round(ci_hi, 6)],
            "crosses_zero": bool(ci_lo <= 0 <= ci_hi),
        }
        r = results[f"w{w}"]
        print(f"  PE S&P600={r['pe_sp600_mean']:.4f}  PE S&P500={r['pe_sp500_mean']:.4f}")
        print(f"  diff (600-500) media={r['diff_mean_all']:+.4f}  "
              f"nonoverlap n={r['n_nonoverlap']} media={r['diff_mean_nonoverlap']:+.4f}")
        print(f"  CI95=[{ci_lo:+.4f}, {ci_hi:+.4f}]  cruza cero: {r['crosses_zero']}")

    with open(OUT_DIR / f"results{SUFFIX}.json", "w") as f:
        json.dump({"config": {"m": M, "tau": TAU, "windows": WINDOWS,
                              "boot_reps": BOOT_REPS, "seed": SEED},
                   "results": results}, f, indent=2)
    print(f"\nResultados -> {OUT_DIR / f'results{SUFFIX}.json'}")


if __name__ == "__main__":
    main()
