"""E8 charts: heatmap de la red ETE con significancia + flujo neto, crisis vs calma,
y serie anual de ETE media con crisis sombreadas."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

NAMES = {"XLB": "Materiales", "XLC": "Comunicaciones", "XLE": "Energía",
         "XLF": "Financiero", "XLI": "Industrial", "XLK": "Tecnología",
         "XLP": "Cons. básico", "XLRE": "Real Estate", "XLU": "Utilities",
         "XLV": "Salud", "XLY": "Cons. discrecional"}
CRISIS_SPANS = [("2000-03-24", "2002-10-09"), ("2007-10-09", "2009-03-09"),
                ("2020-02-19", "2020-03-23"), ("2022-01-03", "2022-10-12")]


def bh_fdr(pvals, alpha=0.05):
    import numpy as _np
    pvals = _np.asarray(pvals)
    m = len(pvals)
    order = _np.argsort(pvals)
    passed = pvals[order] <= alpha * _np.arange(1, m + 1) / m
    k = _np.max(_np.nonzero(passed)[0]) + 1 if passed.any() else 0
    mask = _np.zeros(m, dtype=bool)
    mask[order[:k]] = True
    return mask


def chart_heatmap(subset: list[str] | None = None, suffix: str = "") -> None:
    mat = pd.read_csv(OUT / "q1_ete_matrix.csv", index_col=0)
    with open(OUT / "results.json") as f:
        res = json.load(f)
    if subset is None:
        signif = {tuple(k.split("->")) for k, v in res["q1"]["pairs"].items()
                  if v["significant"]}
        net = pd.Series(res["q1"]["net_flow"]).sort_values()
        title_note = "(* = significativo tras FDR 5%)"
    else:
        # Robustez: FDR y flujo neto recomputados solo sobre los pares del subset
        mat = mat.loc[subset, subset]
        pares = [(s, t) for s in subset for t in subset if s != t]
        pvals = [res["q1"]["pairs"][f"{s}->{t}"]["pval"] for s, t in pares]
        mask = bh_fdr(pvals)
        signif = {p for p, ok in zip(pares, mask) if ok}
        net = pd.Series({t: sum(res["q1"]["pairs"][f"{t}->{o}"]["ete"]
                                for o in subset if o != t)
                            - sum(res["q1"]["pairs"][f"{o}->{t}"]["ete"]
                                  for o in subset if o != t)
                         for t in subset}).sort_values()
        print(f"  subset {len(subset)} sectores: {len(signif)}/{len(pares)} "
              f"significativos post-FDR")
        title_note = (f"solo los {len(subset)} sectores con historia completa "
                      "(* = FDR 5% sobre este subconjunto)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.45, 1]})
    tickers = mat.index.tolist()
    im = ax1.imshow(mat.values.astype(float) * 1000, cmap="viridis")
    labels = [NAMES[t] for t in tickers]
    ax1.set_xticks(range(len(tickers)), labels, rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(range(len(tickers)), labels, fontsize=8)
    ax1.set_xlabel("Receptor")
    ax1.set_ylabel("Emisor")
    for i, s in enumerate(tickers):
        for j, t in enumerate(tickers):
            if i == j:
                continue
            if (s, t) in signif:
                ax1.text(j, i, "*", ha="center", va="center", color="white",
                         fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax1, shrink=0.8, label="ETE (milésimas de bit)")
    ax1.set_title(f"ETE por par dirigido, 1998-2025\n{title_note}")

    colors = ["C3" if v < 0 else "C0" for v in net.values]
    ax2.barh([NAMES[t] for t in net.index], net.values * 1000, color=colors, alpha=0.85)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Flujo neto (out − in, milésimas de bit)")
    ax2.set_title("Emisores y receptores netos")
    ax2.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUT / f"e8_heatmap{suffix}.png", dpi=150)
    plt.close()
    print(f"-> e8_heatmap{suffix}.png")


def chart_crisis() -> None:
    with open(OUT / "results.json") as f:
        res = json.load(f)
    q2 = res["q2"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    means = [q2["mean_ete_calma"] * 1000, q2["mean_ete_crisis"] * 1000]
    cis = [np.array(q2["ci_calma"]) * 1000, np.array(q2["ci_crisis"]) * 1000]
    ax.bar(["Calma", "Crisis"], means, color=["C0", "C3"], alpha=0.85, width=0.55)
    for i, (m, ci) in enumerate(zip(means, cis)):
        ax.errorbar(i, m, yerr=[[m - ci[0]], [ci[1] - m]], color="black",
                    capsize=5, linewidth=1.4)
    for name, ep in q2["episodes"].items():
        if ep.get("mean_ete") is not None:
            ax.scatter(1, ep["mean_ete"] * 1000, marker="D", s=28, color="darkred",
                       zorder=3)
            ax.annotate(name, (1, ep["mean_ete"] * 1000), xytext=(8, 0),
                        textcoords="offset points", fontsize=7, va="center")
    ax.set_ylabel("ETE media de la red (milésimas de bit)")
    ax.set_title("Flujo de información entre sectores: crisis vs calma\n"
                 f"(diferencia media {q2['mean_diff']*1000:+.2f} milésimas, "
                 f"CI 95% [{q2['ci_diff'][0]*1000:+.2f}; {q2['ci_diff'][1]*1000:+.2f}])")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "e8_crisis.png", dpi=150)
    plt.close()
    print("-> e8_crisis.png")


def chart_annual() -> None:
    annual = pd.read_csv(OUT / "annual_ete.csv", index_col="year")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(annual.index, annual["mean_ete"] * 1000, marker="o", markersize=3.5,
            color="C0", linewidth=1.3)
    for a, b in CRISIS_SPANS:
        ax.axvspan(pd.Timestamp(a).year, pd.Timestamp(b).year + 0.3, alpha=0.15,
                   color="red")
    ax.set_xlabel("Año")
    ax.set_ylabel("ETE media de la red (milésimas de bit)")
    ax.set_title("ETE media entre sectores por año calendario (descriptivo)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "e8_anual.png", dpi=150)
    plt.close()
    print("-> e8_anual.png")


FULL_HISTORY_9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

if __name__ == "__main__":
    chart_heatmap()
    chart_heatmap(subset=FULL_HISTORY_9, suffix="_9full")
    chart_crisis()
    chart_annual()
