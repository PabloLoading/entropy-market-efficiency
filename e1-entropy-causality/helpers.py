"""Shared constants, paths and plotting for entropy-causality sub-experiments."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Make the sibling `common/` package importable.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

EXPERIMENT_DIR = _HERE
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
CACHE_DIR = OUTPUTS_DIR / ".cache"
RESULTS_PATH = OUTPUTS_DIR / "results.json"
for d in (CHARTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

SPY_TICKER = "SPY"
SPY_START = "1993-01-29"
SPY_END = "2025-12-31"
INDEX_TICKERS = ["QQQ", "IWM", "DIA"]

PE_M = 3
PE_TAU = 1
PE_WINDOW = 20
N_BOOT = 500
BLOCK_SIZE = 30
EDGE_THRESHOLD = 0.01
SEED = 42

EDGE_COLOR_POS = "#c0392b"
EDGE_COLOR_NEG = "#2980b9"
NODE_COLOR_DEFAULT = "#bdc3c7"
NODE_COLOR_TARGET = "#ff6b6b"
NODE_COLOR_TESTED = "#4ecdc4"


@dataclass
class CausalityResult:
    """Outputs of one DirectLiNGAM run on one factor panel."""
    labels: list[str]
    causal_order: list[str]
    adjacency: np.ndarray
    edge_prob: np.ndarray
    mean_adj: np.ndarray
    non_gaussian: dict[str, bool]
    n_obs: int
    tie_rate: float = 0.0
    diffs: dict[str, int] = None
    bootstrap_params: dict = None
    error_independence: dict = None

    def high_confidence_edges(self, prob_threshold=0.5):
        edges = []
        for i, target in enumerate(self.labels):
            for j, source in enumerate(self.labels):
                if i == j:
                    continue
                if self.edge_prob[i, j] >= prob_threshold:
                    edges.append({
                        "source": source,
                        "target": target,
                        "mean_effect": float(self.mean_adj[i, j]),
                        "edge_prob": float(self.edge_prob[i, j]),
                    })
            edges.sort(key=lambda e: -e["edge_prob"])
        return edges


def draw_causal_graph(ax, title, adjacency, labels, edge_prob=None,
                      target="Returns", tested="Entropy", prob_threshold=0.5):
    """Circular causal graph. If edge_prob given, only edges above threshold are drawn."""
    G = nx.DiGraph()
    G.add_nodes_from(labels)
    for i, dst in enumerate(labels):
        for j, src in enumerate(labels):
            if i == j or abs(adjacency[i, j]) <= EDGE_THRESHOLD:
                continue
            if edge_prob is not None and edge_prob[i, j] < prob_threshold:
                continue
            G.add_edge(src, dst, weight=adjacency[i, j],
                       prob=None if edge_prob is None else edge_prob[i, j])

    angle = 2 * np.pi / len(labels)
    pos = {lab: (np.cos(i * angle + np.pi / 2), np.sin(i * angle + np.pi / 2))
           for i, lab in enumerate(labels)}

    color_map = {target: NODE_COLOR_TARGET, tested: NODE_COLOR_TESTED}
    node_colors = [color_map.get(n, NODE_COLOR_DEFAULT) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2400,
                           edgecolors="black", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    for u, v, d in G.edges(data=True):
        color = EDGE_COLOR_POS if d["weight"] > 0 else EDGE_COLOR_NEG
        width = max(1.2, min(4.5, abs(d["weight"]) * 6))
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=[color],
            arrows=True, arrowstyle="-|>", arrowsize=22, width=width,
            connectionstyle="arc3,rad=0.15",
            min_source_margin=20, min_target_margin=20, ax=ax,
        )
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        lbl = f"{d['weight']:+.3f}"
        if d.get("prob") is not None:
            lbl += f"\n({d['prob']:.0%})"
        edge_labels[(u, v)] = lbl
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7,
                                  bbox=dict(boxstyle="round,pad=0.18",
                                            facecolor="white", alpha=0.85),
                                  ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def save_dual_graph(result_pe, result_wpe, title_pe, title_wpe, out_path,
                    target="Returns", tested="Entropy"):
    """Side-by-side bootstrap-robust causal graphs for PE and WPE variants."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    draw_causal_graph(axes[0], title_pe, result_pe.mean_adj, result_pe.labels,
                      edge_prob=result_pe.edge_prob, target=target, tested=tested)
    draw_causal_graph(axes[1], title_wpe, result_wpe.mean_adj, result_wpe.labels,
                      edge_prob=result_wpe.edge_prob, target=target, tested=tested)
    fig.suptitle(out_path.stem.replace("_", " ").upper(), fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def merge_results(new_entries):
    """Merge a dict of entries into outputs/results.json (creates if missing)."""
    existing = {}
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text())
    existing.update(new_entries)
    RESULTS_PATH.write_text(json.dumps(existing, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"not serializable: {type(o)}")


def result_to_payload(r: CausalityResult):
    return {
        "n_obs": r.n_obs,
        "labels": r.labels,
        "causal_order": r.causal_order,
        "non_gaussian": r.non_gaussian,
        "adjacency": r.adjacency.tolist(),
        "edge_prob": r.edge_prob.tolist(),
        "mean_adj": r.mean_adj.tolist(),
        "high_confidence_edges": r.high_confidence_edges(),
        "tie_rate_log_returns": r.tie_rate,
        "diffs_applied": r.diffs or {},
        "bootstrap_params": r.bootstrap_params or {},
        "error_independence": r.error_independence or {},
    }
