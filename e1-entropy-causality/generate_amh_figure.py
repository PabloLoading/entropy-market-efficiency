"""Generate the AMH bridge figure: position of PE in the causal order across
60 rolling windows on SPY (1993-2025)."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date

from helpers import CHARTS_DIR


def main():
    data = json.load(open("outputs/results.json"))
    rolling = data["e1_rolling_pe"]
    windows = rolling["windows"]

    dates = [date.fromisoformat(w["end_date"]) for w in windows]
    pe_positions = [w["causal_order"].index("Entropy") + 1 for w in windows]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, pe_positions, marker="o", markersize=5,
            color="#4ecdc4", linewidth=1.5)
    ax.axhline(y=1, color="#c0392b", linestyle=":", alpha=0.5, label="ancestor (top)")
    ax.axhline(y=7, color="#2980b9", linestyle=":", alpha=0.5, label="último (bottom)")

    # Shade key crisis periods
    ax.axvspan(date(2000, 3, 1), date(2002, 10, 1), alpha=0.10, color="gray")
    ax.axvspan(date(2007, 10, 1), date(2009, 6, 1), alpha=0.10, color="gray")
    ax.axvspan(date(2020, 2, 1), date(2020, 12, 1), alpha=0.10, color="gray")

    ax.set_ylim(0.5, 7.5)
    ax.invert_yaxis()
    ax.set_yticks(range(1, 8))
    ax.set_yticklabels([f"pos. {i}" for i in range(1, 8)])
    ax.set_xlabel("Fin de la ventana rolling (2 años)")
    ax.set_ylabel("Posición de PE en el orden causal")
    ax.set_title("Evolución temporal de la posición causal de la PE en SPY (1993--2025)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out_path = CHARTS_DIR / "parte1_amh_rolling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
