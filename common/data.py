"""Cached data loaders and standard factor panel for entropy/causality experiments."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .entropy import rolling_perm_entropy

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def load_prices(ticker, start, end, interval="1d", refresh=False):
    """Load OHLCV from yfinance with on-disk parquet cache.

    Args:
        ticker: yfinance ticker (e.g., 'SPY', '^GSPC').
        start, end: 'YYYY-MM-DD' strings.
        interval: yfinance interval string.
        refresh: if True, bypass cache.

    Returns:
        DataFrame with columns Open/High/Low/Close/Adj Close/Volume.
    """
    key = f"{ticker}_{start}_{end}_{interval}".replace("^", "idx_").replace("/", "_")
    cache_path = CACHE_DIR / f"{key}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    raw = yf.download(ticker, start=start, end=end, interval=interval,
                      auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.to_parquet(cache_path)
    return raw


def build_factor_panel(prices, m=3, tau=1, pe_window=20, momentum_window=252,
                        reversal_window=5, vol_window=20, weighted_pe=False,
                        include_ltrevrsl=False, ltrevrsl_lookback=252,
                        ltrevrsl_lag=252, vix_series=None):
    """Build the factor panel used in entropy-causality.

    Base factors: Returns, Entropy, Momentum, Reversal, Volatility, Liquidity,
    Volume. Optional add-ons: LongReversal (Barra LTREVRSL analog), VIX.

    Args:
        prices: DataFrame with 'Adj Close' and 'Volume'.
        m, tau, pe_window: PE parameters.
        weighted_pe: if True, use WPE in the Entropy column.
        momentum_window, reversal_window, vol_window: factor lookback windows.
        include_ltrevrsl: if True, add LongReversal column (cumulative return
            over `ltrevrsl_lookback` days lagged by `ltrevrsl_lag` days, i.e.
            the return over months 12-24 in the past by default).
        ltrevrsl_lookback, ltrevrsl_lag: LongReversal window parameters.
        vix_series: optional pandas Series indexed by date with VIX values.
            When provided, adds a 'VIX' column reindexed to the panel's dates.

    Returns:
        DataFrame of factors aligned to prices' index, NaN-dropped.
    """
    price = prices["Adj Close"].astype(float)
    volume = prices["Volume"].astype(float)

    log_ret = np.log(price / price.shift(1))
    entropy = rolling_perm_entropy(log_ret, m=m, tau=tau, window=pe_window,
                                    weighted=weighted_pe, normalize=True)
    momentum = log_ret.rolling(momentum_window).sum()
    reversal = log_ret.rolling(reversal_window).sum()
    volatility = log_ret.rolling(vol_window).std()
    dollar_volume = price * volume
    liquidity = (log_ret.abs() / dollar_volume).rolling(vol_window).mean()
    log_volume = np.log(volume.replace(0, np.nan))

    columns = {
        "Returns": log_ret,
        "Entropy": entropy,
        "Momentum": momentum,
        "Reversal": reversal,
        "Volatility": volatility,
        "Liquidity": liquidity,
        "Volume": log_volume,
    }

    if include_ltrevrsl:
        columns["LongReversal"] = log_ret.rolling(ltrevrsl_lookback).sum().shift(ltrevrsl_lag)

    if vix_series is not None:
        columns["VIX"] = pd.Series(vix_series).reindex(log_ret.index)

    factors = pd.DataFrame(columns).dropna()
    return factors


def _ffd_weights(d, thres=1e-4, max_k=10000):
    """Recurrence weights for fixed-width fractional differentiation.

    w_0 = 1;  w_k = -w_{k-1} · (d - k + 1) / k.
    Truncated when |w_k| < thres.
    """
    weights = [1.0]
    k = 1
    while k < max_k:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < thres:
            break
        weights.append(w)
        k += 1
    return np.array(weights)


def fractional_difference(series, d=0.3, thres=1e-4):
    """Fixed-width Fractional Differentiation (López de Prado 2018, Ch. 5).

    Retains long-term memory of the series while achieving stationarity,
    unlike integer differentiation (d=1) which destroys all memory. LdP's
    rule of thumb for financial series: d in [0.3, 0.5].

    Args:
        series: 1D pandas Series or numpy array.
        d: fractional differentiation order in (0, 1).
        thres: weight threshold to truncate the lookback window.

    Returns:
        pandas Series aligned to input index; first `lookback` values are NaN.
    """
    x = pd.Series(series) if not isinstance(series, pd.Series) else series
    w = _ffd_weights(d, thres)
    lookback = len(w) - 1
    values = x.values.astype(float)
    n = len(values)
    out = np.full(n, np.nan)
    for t in range(lookback, n):
        window = values[t - lookback : t + 1]
        if np.isnan(window).any():
            continue
        out[t] = float(np.dot(w, window[::-1]))
    return pd.Series(out, index=x.index)


def difference_nonstationary(df, alpha=0.05, method="ffd", d=0.3, thres=1e-4, max_diff=1):
    """Make each column stationary using ADF as decision test.

    Args:
        df: DataFrame of factors.
        alpha: significance level for ADF (H0 = unit root).
        method: "ffd" (Fractional, LdP 2018) or "integer" (first difference).
        d: FFD order (used if method='ffd'). LdP recommends 0.3-0.5.
        thres: FFD weight truncation threshold.
        max_diff: max integer diffs to apply (used if method='integer').

    Returns:
        (df_stationary, dict {col: {"method", "d", "n_diffs"}}).
        Falls back to integer diff if FFD does not achieve stationarity.
    """
    from statsmodels.tsa.stattools import adfuller
    out = df.copy()
    diffs = {}
    for col in out.columns:
        _, p_orig, *_ = adfuller(out[col].dropna())
        if p_orig < alpha:
            diffs[col] = {"method": "none", "d": 0.0, "n_diffs": 0}
            continue
        if method == "ffd":
            ffd = fractional_difference(out[col], d=d, thres=thres)
            _, p_ffd, *_ = adfuller(ffd.dropna())
            if p_ffd < alpha:
                out[col] = ffd
                diffs[col] = {"method": "ffd", "d": float(d), "n_diffs": 0}
                continue
            # FFD failed to achieve stationarity → fall back to integer
            out[col] = out[col].diff()
            diffs[col] = {"method": "integer_fallback", "d": 1.0, "n_diffs": 1}
        else:
            n = 0
            while n < max_diff:
                _, p_val, *_ = adfuller(out[col].dropna())
                if p_val < alpha:
                    break
                out[col] = out[col].diff()
                n += 1
            diffs[col] = {"method": "integer", "d": 1.0, "n_diffs": n}
    out = out.dropna()
    return out, diffs
