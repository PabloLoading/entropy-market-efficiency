"""Permutation entropy (Bandt & Pompe 2002) and weighted permutation entropy
(Fadlallah et al. 2013)."""
from __future__ import annotations
from math import factorial

import numpy as np
import pandas as pd


def perm_entropy(x, m=3, tau=1, weighted=False, normalize=True):
    """Permutation entropy (PE) or weighted permutation entropy (WPE).

    Ties are discarded (windows where np.unique(window).size < m are skipped).

    Args:
        x: 1D array of values.
        m: embedding dimension (>=2).
        tau: time delay (>=1).
        weighted: if True, weight each ordinal pattern by the variance of its
            embedding window (WPE, Fadlallah 2013).
        normalize: if True, divide by log2(m!) so the result lies in [0, 1].

    Returns:
        Entropy in bits (float). NaN if not computable.
    """
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        return float("nan")

    span = (m - 1) * tau
    n_patterns = x.size - span
    if n_patterns <= 0:
        return float("nan")

    windows = np.lib.stride_tricks.sliding_window_view(x, span + 1)[:, ::tau]

    counts: dict[tuple[int, ...], float] = {}
    total = 0.0
    for i in range(n_patterns):
        w = windows[i]
        if np.unique(w).size != m:
            continue
        key = tuple(np.argsort(w, kind="mergesort"))
        weight = float(w.var(ddof=0)) if weighted else 1.0
        counts[key] = counts.get(key, 0.0) + weight
        total += weight

    if total <= 0 or not counts:
        return float("nan")

    p = np.array(list(counts.values()), dtype=float) / total
    h = float(-(p * np.log2(p)).sum())
    if normalize:
        h /= np.log2(factorial(m))
    return h


def rolling_perm_entropy(series, m=3, tau=1, window=20, weighted=False, normalize=True):
    """Rolling permutation entropy over a 1D pandas Series.

    Args:
        series: pandas Series or 1D array.
        window: rolling window length in observations.
        m, tau, weighted, normalize: see perm_entropy.

    Returns:
        pandas Series of entropy values, aligned to the original index.
    """
    s = pd.Series(series)
    return s.rolling(window, min_periods=window).apply(
        lambda w: perm_entropy(w, m=m, tau=tau, weighted=weighted, normalize=normalize),
        raw=True,
    )


def tie_rate(x, m=3, tau=1):
    """Fraction of embedding windows discarded due to ties.

    Useful as a diagnostic when applying PE to discrete or low-precision data.
    """
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        return float("nan")
    span = (m - 1) * tau
    n_patterns = x.size - span
    if n_patterns <= 0:
        return float("nan")
    windows = np.lib.stride_tricks.sliding_window_view(x, span + 1)[:, ::tau]
    n_ties = sum(1 for w in windows if np.unique(w).size != m)
    return n_ties / n_patterns
