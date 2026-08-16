"""Reusable causality pipeline: panel → standardize → ADF/diff → LiNGAM + bootstrap.

One call returns a CausalityResult ready to plot or serialize.
"""
from __future__ import annotations

from pathlib import Path

import helpers  # noqa: F401  — adds ../ to sys.path so `common` is importable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import (
    block_bootstrap_lingam,
    build_factor_panel,
    difference_nonstationary,
    error_independence_summary,
    fit_lingam,
    gaussianity_test,
    load_prices,
    tie_rate,
)
from helpers import (
    BLOCK_SIZE, CACHE_DIR, EDGE_THRESHOLD, N_BOOT,
    PE_M, PE_TAU, PE_WINDOW, SEED, SPY_END, SPY_START,
    CausalityResult,
)


def _cache_path(tag, n_boot, block_size, edge_threshold):
    suffix = f"_nb{n_boot}_bs{block_size}_et{edge_threshold:g}"
    return CACHE_DIR / f"bootstrap_{tag}{suffix}.npz"


def _load_or_run_bootstrap(X, tag, n_boot=N_BOOT, block_size=BLOCK_SIZE,
                            edge_threshold=EDGE_THRESHOLD, refresh=False):
    cache = _cache_path(tag, n_boot, block_size, edge_threshold)
    if cache.exists() and not refresh:
        z = np.load(cache)
        return {"edge_prob": z["edge_prob"], "mean_adj": z["mean_adj"],
                "std_adj": z["std_adj"], "block_size": int(z["block_size"]),
                "n_boot": int(z["n_boot"])}
    out = block_bootstrap_lingam(X, block_size=block_size, n_boot=n_boot,
                                  edge_threshold=edge_threshold, seed=SEED)
    np.savez(cache, edge_prob=out["edge_prob"], mean_adj=out["mean_adj"],
              std_adj=out["std_adj"], block_size=out["block_size"], n_boot=out["n_boot"])
    return out


def build_panel(ticker, start, end, weighted_pe, m=PE_M, tau=PE_TAU, pe_window=PE_WINDOW,
                add_future_return=False, factors_filter=None,
                include_vix=False, include_ltrevrsl=False):
    prices = load_prices(ticker, start, end)
    vix_series = None
    if include_vix:
        vix_prices = load_prices("^VIX", start, end)
        vix_series = vix_prices["Adj Close"].astype(float)
    panel = build_factor_panel(prices, m=m, tau=tau, pe_window=pe_window,
                                weighted_pe=weighted_pe,
                                include_ltrevrsl=include_ltrevrsl,
                                vix_series=vix_series)
    if add_future_return:
        panel["FutureReturn"] = panel["Returns"].shift(-1)
        panel = panel.dropna()
    if factors_filter is not None:
        keep = [c for c in factors_filter if c in panel.columns]
        if add_future_return and "FutureReturn" in panel.columns:
            keep.append("FutureReturn")
        panel = panel[keep]
    return panel


def standardize_and_diff(panel):
    diffed, diffs = difference_nonstationary(panel)
    X = pd.DataFrame(StandardScaler().fit_transform(diffed),
                      columns=diffed.columns, index=diffed.index)
    return X, diffs


def run_causality(ticker, weighted_pe, tag, start=SPY_START, end=SPY_END,
                  add_future_return=False, m=PE_M, tau=PE_TAU, pe_window=PE_WINDOW,
                  block_size=BLOCK_SIZE, n_boot=N_BOOT, edge_threshold=EDGE_THRESHOLD,
                  refresh_cache=False, factors_filter=None,
                  include_vix=False, include_ltrevrsl=False):
    """End-to-end causality run for one ticker and one entropy variant.

    Returns a CausalityResult and the raw factor matrix X (post-scaling).
    """
    panel = build_panel(ticker, start, end, weighted_pe, m=m, tau=tau,
                         pe_window=pe_window, add_future_return=add_future_return,
                         factors_filter=factors_filter,
                         include_vix=include_vix, include_ltrevrsl=include_ltrevrsl)
    X, diffs = standardize_and_diff(panel)
    model, adj, residuals = fit_lingam(X)
    gauss_df = gaussianity_test(residuals, columns=list(X.columns), method="anderson")
    indep_summary = error_independence_summary(model, X)
    boot = _load_or_run_bootstrap(X, tag, n_boot=n_boot, block_size=block_size,
                                    edge_threshold=edge_threshold, refresh=refresh_cache)
    labels = list(X.columns)
    causal_order = [labels[i] for i in model.causal_order_]
    pe_tie_rate = float(tie_rate(panel["Returns"].values, m=m, tau=tau))
    result = CausalityResult(
        labels=labels,
        causal_order=causal_order,
        adjacency=adj,
        edge_prob=boot["edge_prob"],
        mean_adj=boot["mean_adj"],
        non_gaussian={c: bool(v) for c, v in gauss_df["non_gaussian"].items()},
        n_obs=len(X),
        tie_rate=pe_tie_rate,
        diffs=dict(diffs),
        bootstrap_params={"n_boot": n_boot, "block_size": block_size,
                          "edge_threshold": edge_threshold},
        error_independence={
            "n_pairs": indep_summary["n_pairs"],
            "n_independent": indep_summary["n_independent"],
            "fraction_independent": indep_summary["fraction_independent"],
        },
    )
    return result, X
