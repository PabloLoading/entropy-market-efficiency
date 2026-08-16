"""Parte II daily: rolling block-bootstrapped LiNGAM on 20 US-equity assets.

For each (asset, window_size), runs rolling DirectLiNGAM with block bootstrap
and records edge probabilities per window. The primary output is the
distribution of edge probability from Entropy to FutureReturn across
(asset, window, window_size). Cache is saved as npz per (asset, window_size)
so the experiment can be resumed if interrupted.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from parte2_helpers import (
    BLOCK_SIZE_DAILY, DAILY_ASSETS_FLAT, DAILY_END, DAILY_START, DAILY_STEP,
    DAILY_WINDOWS, N_BOOT_DAILY, PARTE2_CACHE, PARTE2_OUTPUTS,
    build_parte2_panel, run_rolling_bootstrap, summarise_pe_to_fr,
)

from common import load_prices


def cache_path(asset, window_size):
    return PARTE2_CACHE / f"daily_{asset}_w{window_size}.pkl"


def run_asset_window(asset, window_size, refresh=False):
    cache = cache_path(asset, window_size)
    if cache.exists() and not refresh:
        with open(cache, "rb") as f:
            return pickle.load(f)

    prices = load_prices(asset, DAILY_START, DAILY_END)
    if len(prices) < window_size + 100:
        print(f"    [{asset}] insufficient history ({len(prices)} rows) for w={window_size}")
        return []

    panel = build_parte2_panel(prices, add_future_return=True)
    print(f"    [{asset}] panel n_obs={len(panel)}, running rolling w={window_size}...")
    results = run_rolling_bootstrap(
        panel, window_size=window_size, step=DAILY_STEP,
        n_boot=N_BOOT_DAILY, block_size=BLOCK_SIZE_DAILY,
    )
    with open(cache, "wb") as f:
        pickle.dump(results, f)
    return results


def main():
    t0 = time.perf_counter()
    print(f"Parte II daily: {len(DAILY_ASSETS_FLAT)} assets x {len(DAILY_WINDOWS)} window sizes")
    print(f"  Assets: {DAILY_ASSETS_FLAT}")
    print(f"  Windows: {DAILY_WINDOWS}  Step: {DAILY_STEP}  N_boot: {N_BOOT_DAILY}")

    summary_rows = []
    for w in DAILY_WINDOWS:
        for asset in DAILY_ASSETS_FLAT:
            t_a = time.perf_counter()
            results = run_asset_window(asset, w)
            if not results:
                continue
            df = summarise_pe_to_fr(results)
            df.insert(0, "asset", asset)
            df.insert(1, "window_size", w)
            summary_rows.append(df)
            n_robust = int(df["is_robust"].sum())
            elapsed = time.perf_counter() - t_a
            print(f"  {asset:5s}  w={w:3d}  windows={len(df):3d}  "
                  f"PE→FR robust in {n_robust}/{len(df)}  ({elapsed:.0f}s)")

    if summary_rows:
        import pandas as pd
        summary = pd.concat(summary_rows, ignore_index=True)
        summary_path = PARTE2_OUTPUTS / "parte2_daily_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n  Saved summary CSV: {summary_path}")

    print(f"\n  Total daily elapsed: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
