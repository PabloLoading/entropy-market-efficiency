"""Causal discovery utilities for time series with autocorrelation.

Wraps DirectLiNGAM (Shimizu et al. 2011) with:
- Non-gaussianity test on residuals (LiNGAM identifiability requirement).
- Moving-block bootstrap that preserves temporal dependence (vs. lingam's
  built-in i.i.d. bootstrap).
- Rolling DirectLiNGAM to study causal structure stability over time.
"""
from __future__ import annotations

import lingam
import numpy as np
import pandas as pd
from scipy import stats


def fit_lingam(X):
    """Fit DirectLiNGAM and return model + adjacency + residuals.

    Args:
        X: pandas DataFrame (n_obs, n_vars). Should be pre-scaled.

    Returns:
        (model, adjacency_matrix, residuals_ndarray).
    """
    model = lingam.DirectLiNGAM()
    model.fit(X)
    adj = model.adjacency_matrix_
    X_arr = np.asarray(X, dtype=float)
    residuals = X_arr - X_arr @ adj.T
    return model, adj, residuals


def error_independence_summary(model, X, alpha=0.05):
    """Test independence of LiNGAM residuals (identifiability assumption).

    DirectLiNGAM assumes the exogenous noise terms of each variable are mutually
    independent. This uses the library's built-in test to check that assumption
    on the fitted residuals.

    Args:
        model: fitted lingam.DirectLiNGAM instance.
        X: the DataFrame the model was fit on (pre-scaled).
        alpha: significance level for the independence test.

    Returns:
        dict with:
            p_values: (n_vars, n_vars) matrix of p-values; upper triangle only
                is meaningful (element [i, j] tests residuals of i vs j).
            n_pairs: number of variable pairs tested.
            n_independent: number of pairs where p > alpha (residuals appear
                mutually independent).
            fraction_independent: n_independent / n_pairs.
    """
    p_matrix = np.asarray(model.get_error_independence_p_values(X), dtype=float)
    n_vars = p_matrix.shape[0]
    i_idx, j_idx = np.triu_indices(n_vars, k=1)
    p_pairs = p_matrix[i_idx, j_idx]
    p_pairs = p_pairs[~np.isnan(p_pairs)]
    n_pairs = int(p_pairs.size)
    n_indep = int(np.sum(p_pairs > alpha))
    return {
        "p_values": p_matrix,
        "n_pairs": n_pairs,
        "n_independent": n_indep,
        "fraction_independent": (n_indep / n_pairs) if n_pairs else float("nan"),
    }


def gaussianity_test(residuals, columns=None, method="anderson", alpha=0.05):
    """Per-variable test of non-gaussianity on LiNGAM residuals.

    LiNGAM is identifiable only when residuals are non-gaussian (Shimizu 2011).
    If any variable fails to reject gaussianity, the causal direction for that
    variable is suspect.

    Args:
        residuals: (n_obs, n_vars) ndarray.
        columns: optional column names.
        method: "anderson" (Anderson-Darling) or "jarque_bera".
        alpha: significance level.

    Returns:
        DataFrame indexed by variable with stat, p_value, non_gaussian.
    """
    rows = []
    n_vars = residuals.shape[1]
    cols = list(columns) if columns is not None else [f"v{j}" for j in range(n_vars)]
    for j in range(n_vars):
        col = residuals[:, j]
        if method == "anderson":
            r = stats.anderson(col, dist="norm")
            stat = float(r.statistic)
            crit_5pct = float(r.critical_values[2])
            non_gauss = stat > crit_5pct
            p_val = float("nan")
        elif method == "jarque_bera":
            stat, p_val = stats.jarque_bera(col)
            stat, p_val = float(stat), float(p_val)
            non_gauss = p_val < alpha
        else:
            raise ValueError(f"unknown method: {method}")
        rows.append({"stat": stat, "p_value": p_val, "non_gaussian": bool(non_gauss)})
    return pd.DataFrame(rows, index=cols)


def block_bootstrap_lingam(X, block_size=30, n_boot=500, edge_threshold=0.01, seed=42):
    """Moving-block bootstrap for DirectLiNGAM preserving temporal dependence.

    Standard i.i.d. bootstrap destroys autocorrelation and underestimates the
    uncertainty of causal edges on financial series. Block bootstrap concatenates
    contiguous blocks sampled with replacement.

    Args:
        X: pandas DataFrame (n_obs, n_vars). Should be pre-scaled.
        block_size: block length in observations (rule of thumb: block > effective
            autocorrelation length; 30-60 for daily financial data).
        n_boot: number of bootstrap replicates.
        edge_threshold: |B[i,j]| > threshold counts as edge for edge_prob.
        seed: random seed.

    Returns:
        dict with:
            edge_prob: (n_vars, n_vars) array of edge frequency.
            mean_adj: (n_vars, n_vars) mean adjacency.
            std_adj: (n_vars, n_vars) std of adjacency.
            block_size, n_boot.
    """
    rng = np.random.default_rng(seed)
    X_arr = np.asarray(X, dtype=float)
    n_obs, n_vars = X_arr.shape
    cols = list(X.columns) if hasattr(X, "columns") else None
    n_blocks = int(np.ceil(n_obs / block_size))
    max_start = n_obs - block_size

    edge_count = np.zeros((n_vars, n_vars))
    mean_adj = np.zeros((n_vars, n_vars))
    all_adj = np.zeros((n_boot, n_vars, n_vars))

    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n_obs]
        X_boot = pd.DataFrame(X_arr[idx], columns=cols) if cols else pd.DataFrame(X_arr[idx])
        m = lingam.DirectLiNGAM()
        m.fit(X_boot)
        adj = m.adjacency_matrix_
        all_adj[b] = adj
        edge_count += (np.abs(adj) > edge_threshold).astype(float)
        mean_adj += adj

    return {
        "edge_prob": edge_count / n_boot,
        "mean_adj": mean_adj / n_boot,
        "std_adj": all_adj.std(axis=0),
        "block_size": block_size,
        "n_boot": n_boot,
    }


def rolling_lingam(X, window=500, step=126, edge_threshold=0.01):
    """Fit DirectLiNGAM on rolling windows to study causal structure stability.

    A core test of the Adaptive Markets Hypothesis (Lo 2004): if markets evolve,
    the causal graph among factors should change over time.

    Args:
        X: pandas DataFrame indexed by date.
        window: window size in observations (e.g., 500 ≈ 2 years of trading days).
        step: stride between windows in observations.
        edge_threshold: |B[i,j]| > threshold for edge presence flag.

    Returns:
        List of dicts {start_date, end_date, adj, causal_order, edge_flags}.
    """
    out = []
    n_obs = len(X)
    cols = list(X.columns)
    for start in range(0, n_obs - window + 1, step):
        end = start + window
        sub = X.iloc[start:end]
        model = lingam.DirectLiNGAM()
        model.fit(sub)
        adj = model.adjacency_matrix_
        out.append({
            "start_date": sub.index[0],
            "end_date": sub.index[-1],
            "adj": adj,
            "causal_order": [cols[i] for i in model.causal_order_],
            "edge_flags": (np.abs(adj) > edge_threshold).astype(int),
        })
    return out


def edge_stability(rolling_results, src, dst, columns):
    """Fraction of rolling windows where edge src -> dst is present.

    Args:
        rolling_results: output of rolling_lingam.
        src, dst: column names.
        columns: list of column names matching adjacency indexing.

    Returns:
        Float in [0, 1].
    """
    if not rolling_results:
        return float("nan")
    i = columns.index(dst)
    j = columns.index(src)
    flags = [r["edge_flags"][i, j] for r in rolling_results]
    return float(np.mean(flags))
