"""Moving-block bootstrap for time series with autocorrelation.

The standard i.i.d. bootstrap (sample observations independently) destroys
temporal dependence and underestimates uncertainty on financial series. The
moving-block bootstrap concatenates contiguous blocks sampled with replacement,
preserving autocorrelation within each block. See López de Prado (2018),
Advances in Financial Machine Learning, on i.i.d. assumption failures in
finance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_indices(n_obs, block_size, n_boot, seed=42):
    """Generate block-bootstrap index arrays (does not touch data).

    Useful when you want to apply the same resampling to multiple aligned arrays
    or to a complex statistic that consumes data lazily.

    Args:
        n_obs: original sample size.
        block_size: block length (>= effective autocorrelation).
        n_boot: number of bootstrap replicates.
        seed: random seed.

    Returns:
        np.ndarray of shape (n_boot, n_obs) with integer indices into [0, n_obs).
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_obs / block_size))
    max_start = n_obs - block_size
    out = np.empty((n_boot, n_obs), dtype=np.int64)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n_obs]
        out[b] = idx
    return out


def block_bootstrap(data, statistic_fn, block_size=30, n_boot=500, seed=42):
    """Apply block bootstrap to a statistic computed on `data`.

    Args:
        data: 1D array, 2D ndarray, or DataFrame. First axis is time.
        statistic_fn: callable that takes a resample (same type/shape as data)
            and returns a scalar or ndarray.
        block_size: block length.
        n_boot: number of replicates.
        seed: random seed.

    Returns:
        np.ndarray of replicates with shape (n_boot, ...).
    """
    is_df = isinstance(data, pd.DataFrame)
    arr = data.to_numpy() if is_df else np.asarray(data)
    n_obs = arr.shape[0]
    indices = block_bootstrap_indices(n_obs, block_size, n_boot, seed)

    replicates = []
    cols = data.columns if is_df else None
    for b in range(n_boot):
        sample_arr = arr[indices[b]]
        sample = pd.DataFrame(sample_arr, columns=cols) if is_df else sample_arr
        replicates.append(statistic_fn(sample))
    return np.array(replicates)


def percentile_ci(replicates, level=0.95, axis=0):
    """Percentile confidence interval from bootstrap replicates.

    Args:
        replicates: array of bootstrap statistics (axis 0 = replicate dimension).
        level: confidence level in (0, 1).
        axis: replicate axis.

    Returns:
        Tuple (lower, upper) of arrays matching the statistic shape.
    """
    alpha = (1 - level) / 2
    return (
        np.quantile(replicates, alpha, axis=axis),
        np.quantile(replicates, 1 - alpha, axis=axis),
    )
