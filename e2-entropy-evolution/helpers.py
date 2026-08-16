"""Shared constants, IO and plotting for E2 (evolución temporal de la eficiencia)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import load_prices, rolling_perm_entropy  # noqa: E402

OUTPUTS = _HERE / "outputs"
CHARTS = OUTPUTS / "charts"
for d in (OUTPUTS, CHARTS):
    d.mkdir(parents=True, exist_ok=True)

INTRADAY_CSV = _HERE.parent / "datasets" / "spy_1min_2008_2021_cleaned.csv"

CRISIS_BANDS = [
    ("Depresión", "1929-09-01", "1932-06-30"),
    ("Dot-com", "2000-03-01", "2002-10-31"),
    ("GFC", "2007-10-01", "2009-03-31"),
    ("COVID", "2020-02-15", "2020-04-30"),
]

DAILY_TICKER = "^GSPC"
DAILY_START = "1928-01-01"
DAILY_END = "2025-12-31"

DAILY_MAIN = {"m": 3, "window": 252}
DAILY_SENS = [{"m": 3, "window": 504}, {"m": 4, "window": 252}, {"m": 4, "window": 504}]

HOURLY_MAIN = {"m": 3, "window": 140}
HOURLY_SENS = [{"m": 3, "window": 70}, {"m": 3, "window": 280}, {"m": 4, "window": 140}]

FIVEMIN_MAIN = {"m": 3, "window": 390}
FIVEMIN_SENS = [{"m": 3, "window": 195}, {"m": 3, "window": 780}, {"m": 4, "window": 390}]


def load_intraday(freq):
    df = pd.read_csv(INTRADAY_CSV, parse_dates=["date"]).set_index("date").sort_index()
    df = df.between_time("07:30", "14:00")
    rule = {"5min": "5min", "1h": "1h"}[freq]
    return df["close"].astype(float).resample(rule).last().dropna()


def compute_pe_grid(price, configs):
    ret = np.log(price / price.shift(1)).dropna()
    return {
        _label(c): rolling_perm_entropy(ret, m=c["m"], tau=1, window=c["window"])
        for c in configs
    }


def _label(cfg):
    return f"m={cfg['m']}, W={cfg['window']}"


def plot_main(pe_series, title, out_path, crisis_bands=CRISIS_BANDS):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(pe_series.index, pe_series.values, color="black", lw=1.1)
    y_max = pe_series.max()
    for name, start, end in crisis_bands:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if e < pe_series.index[0] or s > pe_series.index[-1]:
            continue
        ax.axvspan(s, e, color="grey", alpha=0.25)
        ax.text(s + (e - s) / 2, y_max, name,
                ha="center", va="top", fontsize=8, color="dimgrey")
    ax.set_ylabel("PE (normalizada)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trend(pe_series, title, out_path):
    yearly = pe_series.resample("1YE").mean().dropna()
    x = np.arange(len(yearly))
    slope, intercept = np.polyfit(x, yearly.values, 1)
    fit = slope * x + intercept
    ss_res = np.sum((yearly.values - fit) ** 2)
    ss_tot = np.sum((yearly.values - yearly.values.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(yearly.index, yearly.values, "o-", color="black", lw=1.2, ms=3, label="PE promedio anual")
    ax.plot(yearly.index, fit, "--", color="crimson", lw=1.5,
            label=f"Regresión lineal ($R^2={r2:.2f}$, slope={slope:+.2e}/año)")
    ax.set_ylabel("PE promedio anual")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def mann_kendall(series):
    import pymannkendall as mk
    r = mk.original_test(series.dropna().values)
    return {"trend": r.trend, "p_value": float(r.p),
            "z": float(r.z), "tau": float(r.Tau), "slope": float(r.slope)}


def bai_perron_breaks(series, n_breaks=5):
    import ruptures as rpt
    s = series.dropna()
    algo = rpt.Binseg(model="l2").fit(s.values)
    idx = algo.predict(n_bkps=n_breaks)
    return [s.index[i - 1] for i in idx if 0 < i < len(s)]


def run_subexperiment(name, price, main_cfg, sens_cfg, title_main,
                       mk_freq, br_freq, n_breaks=5):
    grid = compute_pe_grid(price, [main_cfg] + sens_cfg)
    main_pe = grid[_label(main_cfg)].dropna()

    plot_main(main_pe, title_main, CHARTS / f"e2_{name}_main.png")
    plot_trend(main_pe, title_main + " - tendencia anual",
               CHARTS / f"e2_{name}_trend.png")
    pd.DataFrame(grid).to_csv(OUTPUTS / f"e2_{name}.csv")

    results = {"sub_experiment": name, "main": _label(main_cfg), "configs": []}
    for label, series in grid.items():
        s = series.dropna()
        mk = mann_kendall(s.resample(mk_freq).mean())
        breaks = bai_perron_breaks(s.resample(br_freq).mean(), n_breaks=n_breaks)
        results["configs"].append({
            "config": label,
            "mann_kendall": mk,
            "bai_perron_breaks": [b.strftime("%Y-%m") for b in breaks],
        })
    with open(OUTPUTS / f"e2_{name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    main_res = next(c for c in results["configs"] if c["config"] == _label(main_cfg))
    print(f"[{name}] main config: {_label(main_cfg)}")
    print(f"  Mann-Kendall: {main_res['mann_kendall']}")
    print(f"  Bai-Perron breaks: {main_res['bai_perron_breaks']}")
    return grid, results
