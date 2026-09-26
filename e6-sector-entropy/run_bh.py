"""Corrección por multiplicidad (Benjamini-Hochberg al 5%) sobre los rankings anuales de E6.

Post-procesamiento de outputs/percentiles.json y outputs/markets_percentiles.json: la confianza
direccional de cada sector o mercado (masa bootstrap del lado hacia el que apunta) se convierte en
un p-valor bilateral p = 2 * (1 - masa) y se aplica BH sobre las 11 comparaciones sectoriales y las
9 de mercados, por ventana. No recalcula nada del bootstrap. Output: outputs/bh_correction.json
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
ALPHA = 0.05


def bh(pvals: dict, alpha: float) -> set:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    k = 0
    for i, (_, p) in enumerate(items, 1):
        if p <= alpha * i / m:
            k = i
    return {name for name, _ in items[:k]}


def main() -> None:
    result = {"alpha": ALPHA, "p_value": "2 * (1 - confianza_direccional)", "sectores": {}, "mercados": {}}
    for fname, key in (("percentiles.json", "sectores"), ("markets_percentiles.json", "mercados")):
        data = json.load(open(OUT / fname))
        for window, rows in data.items():
            pv = {}
            for name, v in rows.items():
                c = v["confianza_direccional"]
                c = c / 100 if c > 1 else c
                pv[name] = 2 * (1 - c)
            surv = bh(pv, ALPHA)
            result[key][window] = {name: {"p_bilateral": round(p, 4), "senal": rows[name]["senal"],
                                          "bh_5pct": name in surv} for name, p in pv.items()}
            print(f"{key} {window}: sobreviven BH -> {sorted(surv)}")
    json.dump(result, open(OUT / "bh_correction.json", "w"), indent=2, ensure_ascii=False)
    print(f"-> {OUT / 'bh_correction.json'}")


if __name__ == "__main__":
    main()
