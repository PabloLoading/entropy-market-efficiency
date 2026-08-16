"""Shared helpers, constants and panel builder for Parte II (AMH analysis).

Panel Parte II is intentionally distinct from Parte I: 4 factors + FutureReturn
= 5 variables, chosen to allow smaller rolling windows without breaking
LiNGAM statistical requirements.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sibling `common/` importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import (
    block_bootstrap_lingam,
    difference_nonstationary,
    fit_lingam,
    gaussianity_test,
    load_prices,
    rolling_perm_entropy,
)
from sklearn.preprocessing import StandardScaler

# ---- Constants ----

PARTE2_OUTPUTS = _HERE / "outputs"
PARTE2_CHARTS = PARTE2_OUTPUTS / "charts"
PARTE2_CACHE = PARTE2_OUTPUTS / ".cache" / "parte2"
for d in (PARTE2_CHARTS, PARTE2_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# Daily universe: 9 sectors + 3 indices + 8 stocks = 20 assets
DAILY_ASSETS = {
    "sectors": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLB", "XLU", "XLP", "XLY"],
    "indices": ["SPY", "QQQ", "IWM"],
    "stocks": ["TSLA", "NVDA", "NFLX", "MELI", "KO", "WMT", "MSFT", "INTC"],
}
DAILY_ASSETS_FLAT = (
    DAILY_ASSETS["sectors"] + DAILY_ASSETS["indices"] + DAILY_ASSETS["stocks"]
)

# Hourly sub-experiment: 8 assets, including those with highest PE-activity
# in the daily analysis (XLU, IWM, XLF) so we can confirm the daily finding
# extends or breaks at hourly frequency.
HOURLY_ASSETS = ["SPY", "QQQ", "IWM", "XLU", "XLF", "XLK", "XLE", "TSLA"]

# PE parameters (fixed, inherited from Parte I baseline)
PE_M = 3
PE_TAU = 1
PE_WINDOW = 20

# Panel factors used in Parte II (small panel for smaller rolling windows)
PARTE2_FACTORS = ["Returns", "Volatility", "Liquidity", "Entropy"]

# Rolling parameters
DAILY_START = "1998-01-01"
DAILY_END = "2025-12-31"
DAILY_WINDOWS = [200, 400]      # rolling window sizes in trading days
DAILY_STEP = 42                  # ~2 months

# Hourly data is limited by yfinance to ~730 days from today; use a rolling
# window relative to now rather than fixed calendar dates.
from datetime import date, timedelta
_today = date.today()
HOURLY_END = _today.isoformat()
HOURLY_START = (_today - timedelta(days=700)).isoformat()
HOURLY_WINDOW_DAYS = 100          # tuned up from 50 after sensitivity test
HOURLY_STEP_DAYS = 20             # bigger step to avoid oversampling overlapping windows
HOURLY_PE_WINDOW = 20             # tuned up from 10 for stability of ordinal patterns

# Bootstrap
N_BOOT_DAILY = 200
BLOCK_SIZE_DAILY = 30
N_BOOT_HOURLY = 200
BLOCK_SIZE_HOURLY = 30            # ~4 trading days in hours (up from 13)
EDGE_THRESHOLD = 0.01
SEED = 42


def build_parte2_panel(prices, m=PE_M, tau=PE_TAU, pe_window=PE_WINDOW,
                       add_future_return=True):
    """Build the Parte II panel: 4 factors + FutureReturn.

    Factors: Returns, Volatility, Liquidity, Entropy (in that order in causal
    output). Momentum, Reversal, Volume dropped vs Parte I to keep the panel
    small and enable shorter rolling windows.
    """
    price = prices["Adj Close"].astype(float)
    volume = prices["Volume"].astype(float)

    log_ret = np.log(price / price.shift(1))
    entropy = rolling_perm_entropy(log_ret, m=m, tau=tau, window=pe_window,
                                    weighted=False, normalize=True)
    volatility = log_ret.rolling(pe_window).std()
    dollar_volume = price * volume
    liquidity = (log_ret.abs() / dollar_volume).rolling(pe_window).mean()

    panel = pd.DataFrame({
        "Returns": log_ret,
        "Volatility": volatility,
        "Liquidity": liquidity,
        "Entropy": entropy,
    }).dropna()

    if add_future_return:
        panel["FutureReturn"] = panel["Returns"].shift(-1)
        panel = panel.dropna()
    return panel


def run_rolling_bootstrap(panel, window_size, step, n_boot=200,
                          block_size=30, edge_threshold=EDGE_THRESHOLD,
                          seed=SEED, min_obs_required=None):
    """Rolling block-bootstrapped LiNGAM.

    For each rolling window, standardises + FFD-diffs the sub-panel, fits
    DirectLiNGAM, and runs block bootstrap. Returns a list of records per
    window with edge_prob matrix, mean adjacency, labels, and diagnostic info.

    Args:
        panel: DataFrame indexed by date with the Parte II factors.
        window_size: rolling window size in observations.
        step: stride between windows.
        n_boot: bootstrap replicates per window.
        block_size: block length for the block bootstrap.
        edge_threshold: |B[i,j]| threshold for counting an edge in bootstrap.
        seed: base random seed.
        min_obs_required: if given, skip windows where post-diff obs < this.

    Returns:
        list of dicts with keys: start_date, end_date, labels, causal_order,
        adjacency, edge_prob, mean_adj, n_obs, non_gaussian.
    """
    n_total = len(panel)
    results = []
    if min_obs_required is None:
        min_obs_required = max(50, window_size // 3)

    for start in range(0, n_total - window_size + 1, step):
        end = start + window_size
        sub = panel.iloc[start:end]
        try:
            # Integer differencing on rolling windows: FFD's lookback (~387 for
            # d=0.3) is larger than the smallest windows we use here, so it is
            # not viable within the rolling loop. Trade-off is acceptable given
            # the window sizes in Parte II.
            diffed, _ = difference_nonstationary(sub, method="integer")
            if len(diffed) < min_obs_required:
                continue
            X = pd.DataFrame(
                StandardScaler().fit_transform(diffed),
                columns=diffed.columns, index=diffed.index,
            )
            model, adj, residuals = fit_lingam(X)
            gauss = gaussianity_test(residuals, columns=list(X.columns),
                                     method="anderson")
            boot = block_bootstrap_lingam(
                X, block_size=block_size, n_boot=n_boot,
                edge_threshold=edge_threshold, seed=seed + start,
            )
            labels = list(X.columns)
            results.append({
                "start_date": sub.index[0],
                "end_date": sub.index[-1],
                "labels": labels,
                "causal_order": [labels[i] for i in model.causal_order_],
                "adjacency": adj,
                "edge_prob": boot["edge_prob"],
                "mean_adj": boot["mean_adj"],
                "n_obs": len(X),
                "non_gaussian": {c: bool(v)
                                 for c, v in gauss["non_gaussian"].items()},
            })
        except Exception as exc:
            print(f"    skipped window [{sub.index[0]}, {sub.index[-1]}]: {exc}")
    return results


def summarise_pe_to_fr(rolling_results, target="FutureReturn", source="Entropy"):
    """For each rolling window, extract the edge probability of `source → target`.

    Returns a DataFrame indexed by end_date with columns:
    edge_prob, mean_effect, is_robust, n_obs, pe_position.
    """
    rows = []
    for r in rolling_results:
        labels = r["labels"]
        if source not in labels or target not in labels:
            continue
        i_t = labels.index(target)
        j_s = labels.index(source)
        p = float(r["edge_prob"][i_t, j_s])
        eff = float(r["mean_adj"][i_t, j_s])
        pe_pos = r["causal_order"].index(source) + 1
        rows.append({
            "end_date": r["end_date"],
            "edge_prob_pe_to_fr": p,
            "mean_effect_pe_to_fr": eff,
            "is_robust": p >= 0.5,
            "n_obs": r["n_obs"],
            "pe_position": pe_pos,
        })
    if not rows:
        return pd.DataFrame(columns=["end_date", "edge_prob_pe_to_fr",
                                      "mean_effect_pe_to_fr", "is_robust",
                                      "n_obs", "pe_position"])
    df = pd.DataFrame(rows).sort_values("end_date").reset_index(drop=True)
    return df
