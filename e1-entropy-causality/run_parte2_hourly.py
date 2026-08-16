"""Parte II hourly sub-experiment: rolling block-bootstrapped LiNGAM at 1h
frequency on 5 diverse assets over the past 2 years.

Panel and methodology follow the daily version; the only differences are the
data frequency, shorter windows (50 trading days ≈ 325 hours), and the block
size of the bootstrap tuned to the hourly resolution (~2 trading days).
"""
from __future__ import annotations

import pickle
import time

from parte2_helpers import (
    BLOCK_SIZE_HOURLY, HOURLY_ASSETS, HOURLY_END, HOURLY_PE_WINDOW,
    HOURLY_START, HOURLY_STEP_DAYS, HOURLY_WINDOW_DAYS, N_BOOT_HOURLY,
    PARTE2_CACHE, PARTE2_OUTPUTS,
    build_parte2_panel, run_rolling_bootstrap, summarise_pe_to_fr,
)

from common import load_prices

HOURS_PER_DAY = 7  # approximate US market hours


def cache_path_hourly(asset):
    return PARTE2_CACHE / f"hourly_{asset}.pkl"


def run_asset_hourly(asset, refresh=False):
    cache = cache_path_hourly(asset)
    if cache.exists() and not refresh:
        with open(cache, "rb") as f:
            return pickle.load(f)

    prices = load_prices(asset, HOURLY_START, HOURLY_END, interval="1h")
    if len(prices) < 500:
        print(f"    [{asset}] insufficient hourly history ({len(prices)} rows)")
        return []

    panel = build_parte2_panel(prices, pe_window=HOURLY_PE_WINDOW,
                                add_future_return=True)
    window_obs = HOURLY_WINDOW_DAYS * HOURS_PER_DAY
    step_obs = HOURLY_STEP_DAYS * HOURS_PER_DAY
    print(f"    [{asset}] panel n_obs={len(panel)}, window={window_obs}, step={step_obs}")
    results = run_rolling_bootstrap(
        panel, window_size=window_obs, step=step_obs,
        n_boot=N_BOOT_HOURLY, block_size=BLOCK_SIZE_HOURLY,
    )
    with open(cache, "wb") as f:
        pickle.dump(results, f)
    return results


def main():
    t0 = time.perf_counter()
    print(f"Parte II hourly: {len(HOURLY_ASSETS)} assets")
    print(f"  Assets: {HOURLY_ASSETS}")
    print(f"  Window: {HOURLY_WINDOW_DAYS} days (~{HOURLY_WINDOW_DAYS * HOURS_PER_DAY} hourly obs)")
    print(f"  Step: {HOURLY_STEP_DAYS} days  N_boot: {N_BOOT_HOURLY}  block_size: {BLOCK_SIZE_HOURLY} hours")

    summary_rows = []
    for asset in HOURLY_ASSETS:
        t_a = time.perf_counter()
        results = run_asset_hourly(asset)
        if not results:
            continue
        df = summarise_pe_to_fr(results)
        df.insert(0, "asset", asset)
        summary_rows.append(df)
        n_robust = int(df["is_robust"].sum())
        elapsed = time.perf_counter() - t_a
        print(f"  {asset:5s}  windows={len(df):3d}  PE→FR robust in {n_robust}/{len(df)}  ({elapsed:.0f}s)")

    if summary_rows:
        import pandas as pd
        summary = pd.concat(summary_rows, ignore_index=True)
        summary_path = PARTE2_OUTPUTS / "parte2_hourly_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n  Saved summary CSV: {summary_path}")

    print(f"\n  Total hourly elapsed: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
