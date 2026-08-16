"""E5 — Cross-product robustness: subsets × indices × entropy variants.

Extends E3 (multi-index) and E4 (multi-subset) by testing the full grid of
their intersection. For each (subset, ticker, variant) combination, runs the
predictive causality experiment (with FutureReturn) and records whether the
entropy variant is structurally disconnected from the panel.

Grid: 3 subsets × 4 indices × 2 variants (PE, WPE) = 24 configurations.

Pre-registered decision rule:
    A variant's conclusion "structurally disconnected" is declared ROBUST if
    Entropy has no robust edge (>= 50% block bootstrap probability) in at
    least 10 of 12 (subset × index) combinations for that variant.

Rolling LiNGAM is NOT included here (already covered by E1 on baseline).
Block size and PE config are NOT crossed here (already covered by
Sensitivity on baseline). See run_all.py for the full experiment sequence.
"""
from __future__ import annotations

from helpers import (
    INDEX_TICKERS, SPY_END, SPY_START, SPY_TICKER,
    merge_results,
)
from pipeline import run_causality
from run_e4_subsets import SUBSETS

INDEX_START = "1993-01-01"


def _entropy_edges(result):
    return [
        e for e in result.high_confidence_edges()
        if e["source"] == "Entropy" or e["target"] == "Entropy"
    ]


def _causes_of_future_return(result):
    return [
        e for e in result.high_confidence_edges()
        if e["target"] == "FutureReturn"
    ]


def main():
    tickers = [SPY_TICKER] + INDEX_TICKERS
    n_subsets = len(SUBSETS)
    n_tickers = len(tickers)
    n_per_variant = n_subsets * n_tickers
    n_total = n_per_variant * 2

    print(f"E5 — Cross-product: {n_subsets} subsets × {n_tickers} indices × 2 variants = {n_total} configurations")
    print(f"    Pre-registered rule: variant ROBUST if disconnected in >= 10/{n_per_variant} (subset × index) combos.")

    grid: dict[tuple[str, str], dict] = {}

    for subset_name, cfg in SUBSETS.items():
        for ticker in tickers:
            start = SPY_START if ticker == SPY_TICKER else INDEX_START
            print(f"\n  [{subset_name}][{ticker}]  size={cfg['size']}  factors={cfg['factors']}")

            res_pe, _ = run_causality(
                ticker, weighted_pe=False,
                tag=f"e5_pe_{subset_name}_{ticker}",
                start=start, add_future_return=True,
                factors_filter=cfg["factors"],
                include_vix=cfg["include_vix"],
                include_ltrevrsl=cfg["include_ltrevrsl"],
            )
            res_wpe, _ = run_causality(
                ticker, weighted_pe=True,
                tag=f"e5_wpe_{subset_name}_{ticker}",
                start=start, add_future_return=True,
                factors_filter=cfg["factors"],
                include_vix=cfg["include_vix"],
                include_ltrevrsl=cfg["include_ltrevrsl"],
            )

            ee_pe = _entropy_edges(res_pe)
            ee_wpe = _entropy_edges(res_wpe)
            fe_pe = _causes_of_future_return(res_pe)
            fe_wpe = _causes_of_future_return(res_wpe)

            pe_disc = not ee_pe
            wpe_disc = not ee_wpe

            print(f"    PE  n={res_pe.n_obs}   disconnected={pe_disc}   FutureReturn causes: {[e['source'] for e in fe_pe] or 'none'}")
            print(f"    WPE n={res_wpe.n_obs}  disconnected={wpe_disc}  FutureReturn causes: {[e['source'] for e in fe_wpe] or 'none'}")

            grid[(subset_name, ticker)] = {
                "pe_disconnected": pe_disc,
                "wpe_disconnected": wpe_disc,
                "pe_entropy_edges": ee_pe,
                "wpe_entropy_edges": ee_wpe,
                "pe_future_causes": [e["source"] for e in fe_pe],
                "wpe_future_causes": [e["source"] for e in fe_wpe],
                "pe_n_obs": res_pe.n_obs,
                "wpe_n_obs": res_wpe.n_obs,
                "size": cfg["size"],
                "citation": cfg["citation"],
            }

    pe_disc_count = sum(1 for v in grid.values() if v["pe_disconnected"])
    wpe_disc_count = sum(1 for v in grid.values() if v["wpe_disconnected"])
    threshold = 10

    verdict_pe = "ROBUST" if pe_disc_count >= threshold else "FRAGILE"
    verdict_wpe = "ROBUST" if wpe_disc_count >= threshold else "FRAGILE"

    print("\n  " + "=" * 60)
    print(f"  SUMMARY (n_per_variant = {n_per_variant})")
    print("  " + "=" * 60)
    print(f"  PE  disconnected in: {pe_disc_count}/{n_per_variant}   verdict: {verdict_pe}")
    print(f"  WPE disconnected in: {wpe_disc_count}/{n_per_variant}   verdict: {verdict_wpe}")

    # Matrix view of PE variant
    print("\n  PE variant matrix (✓ = disconnected, ✗ = has robust edge):")
    print(f"  {'subset':16s}" + "".join(f"{t:>8s}" for t in tickers))
    for subset_name in SUBSETS:
        row = f"  {subset_name:16s}"
        for ticker in tickers:
            marker = "✓" if grid[(subset_name, ticker)]["pe_disconnected"] else "✗"
            row += f"{marker:>8s}"
        print(row)

    print("\n  WPE variant matrix (✓ = disconnected, ✗ = has robust edge):")
    print(f"  {'subset':16s}" + "".join(f"{t:>8s}" for t in tickers))
    for subset_name in SUBSETS:
        row = f"  {subset_name:16s}"
        for ticker in tickers:
            marker = "✓" if grid[(subset_name, ticker)]["wpe_disconnected"] else "✗"
            row += f"{marker:>8s}"
        print(row)

    merge_results({
        "e5_grid": {f"{sn}__{tk}": v for (sn, tk), v in grid.items()},
        "e5_summary": {
            "n_per_variant": n_per_variant,
            "n_total": n_total,
            "pe_disconnected_count": pe_disc_count,
            "wpe_disconnected_count": wpe_disc_count,
            "pre_registered_threshold": threshold,
            "pe_verdict": verdict_pe,
            "wpe_verdict": verdict_wpe,
        },
    })


if __name__ == "__main__":
    main()
