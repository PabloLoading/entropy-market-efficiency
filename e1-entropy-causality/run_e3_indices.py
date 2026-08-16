"""E3 — Predictive causality on US equity indices other than SPY.

Tests whether the predictive causal structure found on SPY (E2) generalizes
to other major US equity indices that span size and concentration axes:
QQQ (tech-heavy NASDAQ-100), IWM (small caps Russell 2000), DIA (Dow Jones 30).
Each index uses its own listing history. Same 7-factor panel + FutureReturn,
same block bootstrap.
"""
from __future__ import annotations

from helpers import (
    CHARTS_DIR, INDEX_TICKERS, SPY_END,
    merge_results, result_to_payload, save_dual_graph,
)
from pipeline import run_causality

INDEX_START = "1993-01-01"


def main():
    print(f"E3 — Predictive causality on {len(INDEX_TICKERS)} indices ({INDEX_START} to {SPY_END})")

    payload: dict[str, dict] = {}

    for ticker in INDEX_TICKERS:
        print(f"\n  [{ticker}]")
        try:
            res_pe, _ = run_causality(ticker, weighted_pe=False,
                                       tag=f"e3_pe_{ticker}",
                                       start=INDEX_START, add_future_return=True)
            res_wpe, _ = run_causality(ticker, weighted_pe=True,
                                        tag=f"e3_wpe_{ticker}",
                                        start=INDEX_START, add_future_return=True)
        except Exception as exc:
            print(f"    skipped: {exc}")
            continue

        print(f"    PE  n_obs={res_pe.n_obs}  order: {' -> '.join(res_pe.causal_order)}")
        print(f"    WPE n_obs={res_wpe.n_obs}  order: {' -> '.join(res_wpe.causal_order)}")

        for r, tag in [(res_pe, "PE"), (res_wpe, "WPE")]:
            hi = r.high_confidence_edges()
            fut = [e for e in hi if e["target"] == "FutureReturn"]
            if fut:
                print(f"    {tag} robust causes of FutureReturn (prob>=50%):")
                for e in fut:
                    print(f"      {e['source']:11s} effect={e['mean_effect']:+.4f}  prob={e['edge_prob']:.1%}")
            else:
                print(f"    {tag} robust causes of FutureReturn: none")

        save_dual_graph(
            res_pe, res_wpe,
            title_pe=f"{ticker} PE  n={res_pe.n_obs}",
            title_wpe=f"{ticker} WPE n={res_wpe.n_obs}",
            out_path=CHARTS_DIR / f"e3_{ticker}_pe_vs_wpe.png",
            target="FutureReturn",
        )

        payload[f"e3_{ticker}_pe"] = result_to_payload(res_pe)
        payload[f"e3_{ticker}_wpe"] = result_to_payload(res_wpe)

    merge_results(payload)
    print(f"\n  Charts in {CHARTS_DIR}/e3_*.png")


if __name__ == "__main__":
    main()
