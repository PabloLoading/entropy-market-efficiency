"""Chart de la descomposicion direccion vs magnitud (run_decomp.py).

e8_decomp_pares.png: ETE del canal direccion vs canal magnitud, un punto por
par dirigido (110), con la diagonal y el par XLF->XLK marcado.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from gen_charts import NAMES
from run_te import OUT


def chart_pairs() -> None:
    t = pd.read_csv(OUT / "decomp_pairs.csv")
    res = json.load(open(OUT / "decomp_results.json"))
    x, y = t.ete_dir * 1000, t.ete_mag * 1000
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sig = t.sig_mag.values
    ax.scatter(x[~sig], y[~sig], s=34, color="grey", alpha=0.7, label="magnitud no significativa")
    ax.scatter(x[sig], y[sig], s=34, color="C0", alpha=0.8, label="magnitud significativa (FDR 5%)")
    lim = max(y.max(), x.max()) * 1.08
    lo = min(x.min(), y.min(), 0) * 1.15
    ax.plot([lo, lim], [lo, lim], color="black", linewidth=0.8, linestyle="--", label="magnitud = dirección")
    ax.axhline(0, color="black", linewidth=0.5); ax.axvline(0, color="black", linewidth=0.5)
    m = t[(t.src == "XLF") & (t.tgt == "XLK")].iloc[0]
    ax.scatter([m.ete_dir * 1000], [m.ete_mag * 1000], s=120, facecolor="none", edgecolor="C3", linewidth=1.8)
    ax.annotate("Financiero → Tecnología", (m.ete_dir * 1000, m.ete_mag * 1000),
                xytext=(10, -14), textcoords="offset points", fontsize=9, color="C3")
    a = res["all_110"]
    ax.set_xlabel("ETE canal dirección (signo del retorno), milésimas de bit")
    ax.set_ylabel("ETE canal magnitud (|retorno| > mediana), milésimas de bit")
    ax.set_title("Qué viaja entre sectores: magnitud vs. dirección, 110 pares, 1998-2025\n"
                 f"dirección: {a['dir']['n_signif']} pares significativos · magnitud: {a['mag']['n_signif']} · "
                 f"diferencia media {a['diff_mag_minus_dir']['mean'] * 1000:+.2f} mils, "
                 f"CI [{a['diff_mag_minus_dir']['ci'][0] * 1000:.2f}; {a['diff_mag_minus_dir']['ci'][1] * 1000:.2f}]",
                 fontsize=9.5)
    ax.set_xlim(lo, lim); ax.set_ylim(lo, lim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT / "e8_decomp_pares.png", dpi=150)
    plt.close()
    print("-> e8_decomp_pares.png")


if __name__ == "__main__":
    chart_pairs()
