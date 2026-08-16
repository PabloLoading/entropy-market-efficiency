"""Shared constants, event detection and IO for E3 (entropía en crashes)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import load_prices, rolling_perm_entropy  # noqa: E402

OUTPUTS = _HERE / "outputs"
CHARTS = OUTPUTS / "charts"
for d in (OUTPUTS, CHARTS):
    d.mkdir(parents=True, exist_ok=True)

TICKER = "^GSPC"
START = "1928-01-01"
END = "2025-12-31"

ASSETS_H2 = [
    {"ticker": "^GSPC", "start": "1928-01-01"},
    {"ticker": "^DJI",  "start": "1928-01-01"},
    {"ticker": "^IXIC", "start": "1971-02-05"},
    {"ticker": "^NDX",  "start": "1985-01-01"},
    {"ticker": "^RUT",  "start": "1987-09-10"},
    {"ticker": "XLK",   "start": "1998-12-22"},
    {"ticker": "XLF",   "start": "1998-12-22"},
    {"ticker": "XLE",   "start": "1998-12-22"},
    {"ticker": "XLU",   "start": "1998-12-22"},
    {"ticker": "XLP",   "start": "1998-12-22"},
    {"ticker": "XLY",   "start": "1998-12-22"},
    {"ticker": "XLI",   "start": "1998-12-22"},
    {"ticker": "XLV",   "start": "1998-12-22"},
    {"ticker": "XLB",   "start": "1998-12-22"},
]
MIN_EVENT_DAYS_H2 = 15

PE_MAIN = {"m": 3, "window": 60}
PE_SENS = [{"m": 3, "window": 180}]

EVENT_MIN_DRAWDOWN = 0.05
EVENT_MAX_DAYS = 252
MIN_EVENT_DAYS_MAIN = 30
MIN_EVENT_DAYS_SENS = [60]
CRASH_CUTOFF = 0.15


def detect_events(close, min_dd=EVENT_MIN_DRAWDOWN, max_days=EVENT_MAX_DAYS):
    dates = close.index
    p = close.values
    n = len(p)
    events = []

    peak_price = p[0]
    peak_idx = 0
    i = 1
    while i < n:
        if p[i] > peak_price:
            peak_price = p[i]
            peak_idx = i
            i += 1
            continue
        if p[i] / peak_price - 1 <= -min_dd:
            trough_idx = i
            trough_price = p[i]
            j = i + 1
            forced = False
            while j < n:
                if p[j] < trough_price:
                    trough_price = p[j]
                    trough_idx = j
                if p[j] >= peak_price:
                    break
                if j - peak_idx >= max_days:
                    forced = True
                    break
                j += 1
            recovery_idx = j if j < n else n - 1
            events.append({
                "peak_date": dates[peak_idx],
                "trough_date": dates[trough_idx],
                "recovery_date": dates[recovery_idx],
                "peak_price": float(peak_price),
                "trough_price": float(trough_price),
                "drawdown": (trough_price - peak_price) / peak_price,
                "duration_days": int((dates[trough_idx] - dates[peak_idx]).days),
                "forced_close": forced,
            })
            i = recovery_idx + 1
            if i < n:
                peak_price = p[recovery_idx]
                peak_idx = recovery_idx
            continue
        i += 1
    return pd.DataFrame(events)


def load_price_ret_pe(pe_window, m, ticker=TICKER, start=START, end=END):
    prices = load_prices(ticker, start, end, interval="1d")
    close = prices["Adj Close"].astype(float).dropna()
    log_ret = np.log(close / close.shift(1)).dropna()
    pe = rolling_perm_entropy(log_ret, m=m, tau=1, window=pe_window).dropna()
    return close, log_ret, pe


def compute_event_metrics(events, log_ret, pe, pe_window,
                           min_event_days=MIN_EVENT_DAYS_MAIN):
    rows = []
    for _, ev in events.iterrows():
        peak, trough = ev["peak_date"], ev["trough_date"]
        if (trough - peak).days < min_event_days:
            continue
        pe_before = pe.loc[:peak]
        if len(pe_before) < pe_window:
            continue
        pe_pre_peak = pe_before.iloc[-pe_window:].mean()

        pe_during = pe.loc[peak:trough]
        if len(pe_during) < 5:
            continue
        pe_min_during = pe_during.min()

        vol_before = log_ret.loc[:peak]
        if len(vol_before) < pe_window:
            continue
        vol_pre_peak = vol_before.iloc[-pe_window:].std()

        rows.append({
            "peak_date": peak,
            "trough_date": trough,
            "drawdown": ev["drawdown"],
            "duration": ev["duration_days"],
            "delta_pe": pe_pre_peak - pe_min_during,
            "pe_pre_peak": pe_pre_peak,
            "vol_pre_peak": vol_pre_peak,
        })
    df = pd.DataFrame(rows)
    df["decade"] = df["peak_date"].dt.year // 10 * 10
    return df


def classify(dd, cutoff=CRASH_CUTOFF):
    return np.where(np.abs(dd) >= cutoff, "crash", "pullback")
