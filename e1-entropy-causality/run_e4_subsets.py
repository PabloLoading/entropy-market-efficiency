"""E4 — Robustness of PE causality across pre-registered factor subsets on SPY.

Three subsets grounded in independent modern heavyweight traditions of asset
pricing / factor investing are tested for the predictive causality question
(does any factor at time t cause the return at t+1?). All subsets include
Returns, Volatility and PE by construction; the remaining factors differ.

Pre-registered subsets:
  A (4 factors): Returns, Volatility, VIX, PE.
      Bollerslev, Patton & Quaedvlieg (2020), "Realized semibetas", JFE.
  B (6 factors): Returns, Volatility, Momentum, Reversal, Volume, PE.
      Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning", RFS.
  C (8 factors): Returns, Volatility, Momentum, Reversal, LongReversal,
                 Liquidity, Volume, PE.
      López de Prado & Zoonekynd (2024), "Correcting the Factor Mirage", JPM.

Pre-registered decision rule: PE is considered "structurally disconnected" in
a subset if it has no robust edge (>= 50% block bootstrap probability) with any
other factor. The overall conclusion is robust if PE is disconnected in at
least 2 of the 3 subsets under the PE variant. WPE results are reported as
metric-robustness check.
"""
from __future__ import annotations

from helpers import (
    CHARTS_DIR, SPY_END, SPY_START, SPY_TICKER,
    merge_results, result_to_payload, save_dual_graph,
)
from pipeline import run_causality


SUBSETS = {
    "A_bollerslev": {
        "factors": ["Returns", "Volatility", "VIX", "Entropy"],
        "size": 4,
        "citation": "Bollerslev, Patton & Quaedvlieg (2020), JFE",
        "tradition": "Realized volatility / variance risk premium",
        "include_vix": True,
        "include_ltrevrsl": False,
    },
    "B_kelly": {
        "factors": ["Returns", "Volatility", "Momentum", "Reversal", "Volume", "Entropy"],
        "size": 6,
        "citation": "Gu, Kelly & Xiu (2020), RFS",
        "tradition": "Machine learning asset pricing",
        "include_vix": False,
        "include_ltrevrsl": False,
    },
    "C_lopezdeprado": {
        "factors": ["Returns", "Volatility", "Momentum", "Reversal", "LongReversal",
                    "Liquidity", "Volume", "Entropy"],
        "size": 8,
        "citation": "López de Prado & Zoonekynd (2024), JPM",
        "tradition": "Causal factor investing (Barra USE4L adapted)",
        "include_vix": False,
        "include_ltrevrsl": True,
    },
}


def _entropy_edges(result):
    return [
        e for e in result.high_confidence_edges()
        if e["source"] == "Entropy" or e["target"] == "Entropy"
    ]


def main():
    print(f"E4 — Factor subsets on {SPY_TICKER} ({SPY_START} to {SPY_END})")
    payload: dict[str, dict] = {}
    disconnected_pe = 0
    disconnected_wpe = 0

    for name, cfg in SUBSETS.items():
        print(f"\n  [{name}]  size={cfg['size']}  {cfg['citation']}")
        print(f"    tradition: {cfg['tradition']}")
        print(f"    factors:   {cfg['factors']}")

        res_pe, _ = run_causality(
            SPY_TICKER, weighted_pe=False, tag=f"e4_pe_{name}",
            add_future_return=True, factors_filter=cfg["factors"],
            include_vix=cfg["include_vix"], include_ltrevrsl=cfg["include_ltrevrsl"],
        )
        res_wpe, _ = run_causality(
            SPY_TICKER, weighted_pe=True, tag=f"e4_wpe_{name}",
            add_future_return=True, factors_filter=cfg["factors"],
            include_vix=cfg["include_vix"], include_ltrevrsl=cfg["include_ltrevrsl"],
        )

        print(f"    PE  n_obs={res_pe.n_obs}   order: {' -> '.join(res_pe.causal_order)}")
        print(f"    WPE n_obs={res_wpe.n_obs}  order: {' -> '.join(res_wpe.causal_order)}")

        entropy_edges_pe = _entropy_edges(res_pe)
        entropy_edges_wpe = _entropy_edges(res_wpe)
        future_edges_pe = [e for e in res_pe.high_confidence_edges() if e["target"] == "FutureReturn"]
        future_edges_wpe = [e for e in res_wpe.high_confidence_edges() if e["target"] == "FutureReturn"]

        if not entropy_edges_pe:
            disconnected_pe += 1
        if not entropy_edges_wpe:
            disconnected_wpe += 1

        print(f"    PE  Entropy robust edges: {entropy_edges_pe or 'none — DISCONNECTED'}")
        print(f"    WPE Entropy robust edges: {entropy_edges_wpe or 'none — DISCONNECTED'}")
        print(f"    PE  robust causes of FutureReturn: {future_edges_pe or 'none'}")
        print(f"    WPE robust causes of FutureReturn: {future_edges_wpe or 'none'}")

        save_dual_graph(
            res_pe, res_wpe,
            title_pe=f"Subset {name}  PE  ({cfg['size']} factores, n={res_pe.n_obs})",
            title_wpe=f"Subset {name}  WPE  ({cfg['size']} factores, n={res_wpe.n_obs})",
            out_path=CHARTS_DIR / f"e4_{name}_pe_vs_wpe.png",
            target="FutureReturn",
        )

        payload[f"e4_{name}_pe"] = {
            **result_to_payload(res_pe),
            "citation": cfg["citation"],
            "tradition": cfg["tradition"],
            "size": cfg["size"],
            "entropy_disconnected": not entropy_edges_pe,
        }
        payload[f"e4_{name}_wpe"] = {
            **result_to_payload(res_wpe),
            "citation": cfg["citation"],
            "tradition": cfg["tradition"],
            "size": cfg["size"],
            "entropy_disconnected": not entropy_edges_wpe,
        }

    n = len(SUBSETS)
    verdict_pe = "ROBUST" if disconnected_pe >= 2 else "FRAGILE"
    verdict_wpe = "ROBUST" if disconnected_wpe >= 2 else "FRAGILE"
    print(f"\n  Summary (pre-registered rule: robust if disconnected in >= 2/{n}):")
    print(f"    PE  variant:  disconnected in {disconnected_pe}/{n}  ->  {verdict_pe}")
    print(f"    WPE variant:  disconnected in {disconnected_wpe}/{n}  ->  {verdict_wpe}")

    payload["e4_summary"] = {
        "n_subsets": n,
        "pe_disconnected_count": disconnected_pe,
        "wpe_disconnected_count": disconnected_wpe,
        "pre_registered_threshold": 2,
        "pe_verdict": verdict_pe,
        "wpe_verdict": verdict_wpe,
    }

    merge_results(payload)
    print(f"\n  Charts in {CHARTS_DIR}/e4_*.png")


if __name__ == "__main__":
    main()
