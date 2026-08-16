"""Sensitivity checks for E1 baseline on SPY.

Two axes are tested independently:
  - PE parameters (m, tau, w) ∈ {(3,1,20), (3,1,60), (4,1,120)}
  - Block bootstrap block_size ∈ {15, 30, 60, 90}

If conclusions hold across both axes, the causal structure is robust to the
two parameter choices most likely to affect it.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from helpers import (
    CHARTS_DIR, SPY_END, SPY_START, SPY_TICKER,
    draw_causal_graph, merge_results,
)
from pipeline import run_causality

PE_CONFIGS = [(3, 1, 20), (3, 1, 60), (4, 1, 120)]
BLOCK_SIZES = [15, 30, 60, 90]


def main():
    print(f"Sensitivity on {SPY_TICKER} ({SPY_START} to {SPY_END})")

    # --- PE configuration sensitivity ---
    print("\n[PE config] (m, tau, w)")
    pe_results = []
    for m, tau, w in PE_CONFIGS:
        label = f"m={m}, tau={tau}, w={w}"
        res, _ = run_causality(SPY_TICKER, weighted_pe=False,
                                tag=f"sens_pe_m{m}_w{w}",
                                m=m, tau=tau, pe_window=w)
        print(f"  {label}: {' -> '.join(res.causal_order)}")
        pe_results.append((label, res))

    fig, axes = plt.subplots(1, len(pe_results), figsize=(8 * len(pe_results), 8))
    for ax, (label, res) in zip(axes, pe_results):
        draw_causal_graph(ax, f"PE {label}\nn={res.n_obs}",
                          res.mean_adj, res.labels, edge_prob=res.edge_prob)
    fig.suptitle("Sensibilidad a parámetros de PE", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    pe_path = CHARTS_DIR / "sensitivity_pe.png"
    fig.savefig(pe_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {pe_path}")

    # --- Block bootstrap size sensitivity ---
    print(f"\n[block_size] ∈ {BLOCK_SIZES}")
    block_results = []
    for block_size in BLOCK_SIZES:
        res, _ = run_causality(SPY_TICKER, weighted_pe=False,
                                tag=f"sens_block_{block_size}",
                                block_size=block_size)
        hc = res.high_confidence_edges(prob_threshold=0.5)
        print(f"  block={block_size}: {len(hc)} robust edges (prob>=50%)")
        for e in hc:
            print(f"    {e['source']:11s} -> {e['target']:11s}  effect={e['mean_effect']:+.4f}  prob={e['edge_prob']:.1%}")
        block_results.append((block_size, res, hc))

    fig, axes = plt.subplots(1, len(block_results), figsize=(7 * len(block_results), 7))
    for ax, (bs, res, _) in zip(axes, block_results):
        draw_causal_graph(ax, f"block={bs}\nn={res.n_obs}",
                          res.mean_adj, res.labels, edge_prob=res.edge_prob)
    fig.suptitle(f"Sensibilidad al block_size del bootstrap (n_boot={res.bootstrap_params['n_boot']})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    block_path = CHARTS_DIR / "sensitivity_block_size.png"
    fig.savefig(block_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {block_path}")

    # Stability summary: which edges survive every block_size at >=50%?
    edge_robust_in_all = {}
    if block_results:
        labels = block_results[0][1].labels
        for i, dst in enumerate(labels):
            for j, src in enumerate(labels):
                if i == j:
                    continue
                probs = [br[1].edge_prob[i, j] for br in block_results]
                if all(p >= 0.5 for p in probs):
                    edge_robust_in_all[f"{src} -> {dst}"] = [float(p) for p in probs]
        print(f"\n  Edges robust across ALL block sizes: {len(edge_robust_in_all)}")
        for k, v in edge_robust_in_all.items():
            print(f"    {k}: probs={['%.0f%%' % (p*100) for p in v]}")

    merge_results({
        "sensitivity_pe": [
            {"label": lab, "causal_order": r.causal_order, "n_obs": r.n_obs}
            for lab, r in pe_results
        ],
        "sensitivity_block_size": [
            {"block_size": bs, "causal_order": r.causal_order,
             "n_obs": r.n_obs, "n_robust_edges": len(hc),
             "robust_edges": hc}
            for bs, r, hc in block_results
        ],
        "sensitivity_block_robust_in_all": edge_robust_in_all,
    })


if __name__ == "__main__":
    main()
