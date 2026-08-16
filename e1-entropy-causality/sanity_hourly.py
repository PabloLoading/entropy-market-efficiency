"""Sensitivity check for hourly: test if larger window and pe_window reveal
causal edges that the baseline (50d/pe_window=10) misses."""
from __future__ import annotations

import time

from parte2_helpers import (
    HOURLY_END, HOURLY_START, build_parte2_panel, run_rolling_bootstrap,
    summarise_pe_to_fr,
)

from common import load_prices


HOURS_PER_DAY = 7

# More generous parameters for sanity check
PE_WINDOW = 20                  # was 10
ROLLING_WINDOW_DAYS = 100       # was 50 (→ ~700 hourly obs per window)
STEP_DAYS = 20                  # slightly larger step so it doesn't take forever
BLOCK_SIZE_HOURS = 30           # was 13 (larger block captures more autocorr)
N_BOOT = 200

TEST_ASSETS = ["SPY", "XLK", "TSLA"]


def main():
    print(f"Sanity hourly with tighter params:")
    print(f"  pe_window={PE_WINDOW}, rolling_window={ROLLING_WINDOW_DAYS}d, step={STEP_DAYS}d, block={BLOCK_SIZE_HOURS}h")

    window_obs = ROLLING_WINDOW_DAYS * HOURS_PER_DAY
    step_obs = STEP_DAYS * HOURS_PER_DAY

    for asset in TEST_ASSETS:
        t0 = time.perf_counter()
        prices = load_prices(asset, HOURLY_START, HOURLY_END, interval="1h")
        panel = build_parte2_panel(prices, pe_window=PE_WINDOW, add_future_return=True)
        print(f"\n[{asset}] panel n_obs={len(panel)}, window={window_obs}, step={step_obs}")
        results = run_rolling_bootstrap(
            panel, window_size=window_obs, step=step_obs,
            n_boot=N_BOOT, block_size=BLOCK_SIZE_HOURS,
        )

        # Full source-to-FR breakdown
        for source in ["Returns", "Volatility", "Liquidity", "Entropy"]:
            probs = []
            for r in results:
                if "FutureReturn" not in r["labels"] or source not in r["labels"]:
                    continue
                fr = r["labels"].index("FutureReturn")
                s = r["labels"].index(source)
                probs.append(float(r["edge_prob"][fr, s]))
            if probs:
                import numpy as np
                probs = np.array(probs)
                n_robust = int((probs >= 0.5).sum())
                print(f"  {source:11s}: mean={probs.mean():.2f}  max={probs.max():.2f}  "
                      f"robust={n_robust}/{len(probs)}")
        elapsed = time.perf_counter() - t0
        print(f"  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
