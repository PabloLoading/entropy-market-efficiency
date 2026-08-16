"""E2 — Predictive causality on SPY (1993-2025).

Adds FutureReturn = Returns shifted by -1 to the factor panel and asks whether
any factor causally precedes next-day returns. Runs both PE and WPE variants.
"""
from __future__ import annotations

from helpers import (
    CHARTS_DIR, SPY_END, SPY_START, SPY_TICKER,
    merge_results, result_to_payload, save_dual_graph,
)
from pipeline import run_causality


def main():
    print(f"E2 — Predictive causality on {SPY_TICKER} ({SPY_START} to {SPY_END})")

    res_pe, _ = run_causality(SPY_TICKER, weighted_pe=False, tag="e2_pe",
                              add_future_return=True)
    res_wpe, _ = run_causality(SPY_TICKER, weighted_pe=True, tag="e2_wpe",
                                add_future_return=True)

    print(f"  PE  n_obs={res_pe.n_obs}  order: {' -> '.join(res_pe.causal_order)}")
    print(f"  WPE n_obs={res_wpe.n_obs}  order: {' -> '.join(res_wpe.causal_order)}")

    save_dual_graph(
        res_pe, res_wpe,
        title_pe=f"PE  (m=3, w=20) + FutureReturn  n={res_pe.n_obs}",
        title_wpe=f"WPE (m=3, w=20) + FutureReturn  n={res_wpe.n_obs}",
        out_path=CHARTS_DIR / "e2_predictive_pe_vs_wpe.png",
        target="FutureReturn",
    )

    for r, tag in [(res_pe, "PE"), (res_wpe, "WPE")]:
        hi = r.high_confidence_edges()
        future_edges = [e for e in hi if e["target"] == "FutureReturn"]
        print(f"  {tag} robust causes of FutureReturn (prob>=50%): {future_edges or 'none'}")

    merge_results({
        "e2_pe": result_to_payload(res_pe),
        "e2_wpe": result_to_payload(res_wpe),
    })
    print(f"  -> {CHARTS_DIR / 'e2_predictive_pe_vs_wpe.png'}")


if __name__ == "__main__":
    main()
