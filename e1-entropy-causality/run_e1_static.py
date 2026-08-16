"""E1 — Static causality on SPY (1993-2025).

Tests whether Entropy is a causal ancestor of Returns, a collider, or a
disconnected factor, using DirectLiNGAM with block bootstrap. Runs the analysis
twice (PE and WPE) and saves a side-by-side graph plus rolling LiNGAM as an
AMH check.
"""
from __future__ import annotations

from helpers import (
    CHARTS_DIR, SPY_END, SPY_START, SPY_TICKER,
    merge_results, result_to_payload, save_dual_graph,
)
from pipeline import run_causality

from common import rolling_lingam

ROLLING_WINDOW = 500
ROLLING_STEP = 126


def main():
    print(f"E1 — Static causality on {SPY_TICKER} ({SPY_START} to {SPY_END})")

    res_pe, X_pe = run_causality(SPY_TICKER, weighted_pe=False, tag="e1_pe")
    res_wpe, _ = run_causality(SPY_TICKER, weighted_pe=True, tag="e1_wpe")

    print(f"  PE  n_obs={res_pe.n_obs}  order: {' -> '.join(res_pe.causal_order)}")
    print(f"  WPE n_obs={res_wpe.n_obs}  order: {' -> '.join(res_wpe.causal_order)}")
    print(f"  PE  non-gauss residuals: {sum(res_pe.non_gaussian.values())}/{len(res_pe.non_gaussian)}")
    print(f"  WPE non-gauss residuals: {sum(res_wpe.non_gaussian.values())}/{len(res_wpe.non_gaussian)}")

    save_dual_graph(
        res_pe, res_wpe,
        title_pe=f"PE  (m=3, w=20)  n={res_pe.n_obs}",
        title_wpe=f"WPE (m=3, w=20)  n={res_wpe.n_obs}",
        out_path=CHARTS_DIR / "e1_static_pe_vs_wpe.png",
    )

    print(f"  Rolling LiNGAM on PE (window={ROLLING_WINDOW}, step={ROLLING_STEP})...")
    rolling = rolling_lingam(X_pe, window=ROLLING_WINDOW, step=ROLLING_STEP)
    rolling_summary = []
    for r in rolling:
        rolling_summary.append({
            "start_date": str(r["start_date"].date()),
            "end_date": str(r["end_date"].date()),
            "causal_order": r["causal_order"],
        })
    print(f"  {len(rolling_summary)} rolling windows fitted")

    merge_results({
        "e1_pe": result_to_payload(res_pe),
        "e1_wpe": result_to_payload(res_wpe),
        "e1_rolling_pe": {
            "window": ROLLING_WINDOW,
            "step": ROLLING_STEP,
            "n_windows": len(rolling_summary),
            "windows": rolling_summary,
        },
    })
    print(f"  -> {CHARTS_DIR / 'e1_static_pe_vs_wpe.png'}")


if __name__ == "__main__":
    main()
