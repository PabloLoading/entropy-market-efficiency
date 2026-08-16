"""E8: effective transfer entropy (ETE) entre los 11 sectores SPDR.

Config unica: 3 estados por terciles, k=l=1, TE en bits.
- Q1: ETE por par dirigido sobre el periodo completo + p-valor por block bootstrap
      de la fuente + FDR (Benjamini-Hochberg 5%) + flujo neto por sector.
- Q1b: ETE media de la red por anio calendario (descriptivo, terciles por anio).
- Q2: crisis (union de los 4 episodios de 5.6) vs calma, terciles por muestra;
      diferencia pareada por par con CI por bootstrap sobre pares.

Outputs en outputs/: results.json, q1_ete_matrix.csv, q1_pvalues.csv,
q1_net_flow.csv, annual_ete.csv.
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "e6-sector-entropy" / "data"  # reusa los parquets del e6
OUT = HERE / "outputs"

TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
CRISIS = {
    "dot-com": ("2000-03-24", "2002-10-09"),
    "crisis-2008": ("2007-10-09", "2009-03-09"),
    "covid": ("2020-02-19", "2020-03-23"),
    "tasas-2022": ("2022-01-03", "2022-10-12"),
}
MIN_EPISODE_OBS = 150   # por debajo, el desglose por episodio no se reporta (COVID ~23 dias)
MIN_YEAR_OBS = 200      # dias comunes minimos para la vista anual de un par

SEED = 42
B_BIAS = 100    # barajadas para la correccion effective
R_NULL = 1000   # replicas del block bootstrap para el p-valor
BLOCK = 20      # dias por bloque en la nula
ALPHA_FDR = 0.05
R_PAIRS = 2000  # bootstrap sobre pares para los CI de Q2


def load_returns() -> pd.DataFrame:
    series = {}
    for t in TICKERS:
        px = pd.read_parquet(DATA / f"{t}.parquet")["Adj Close"].dropna()
        series[t] = np.log(px / px.shift(1)).dropna()
    return pd.DataFrame(series)  # NaN donde el ETF no existia


def terciles(x: np.ndarray) -> np.ndarray:
    """Estados 0/1/2 por terciles de la propia muestra."""
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    return np.digitize(x, [q1, q2])


def te_bits(sx: np.ndarray, sy: np.ndarray) -> float:
    """TE(Y->X) plug-in en bits, k=l=1, sobre series de estados alineadas."""
    idx = sx[1:] * 9 + sx[:-1] * 3 + sy[:-1]
    joint = np.bincount(idx, minlength=27).astype(float).reshape(3, 3, 3)
    n = joint.sum()
    p_xyz = joint / n                          # p(next, xnow, ynow)
    p_ctx = p_xyz.sum(axis=0, keepdims=True)   # p(xnow, ynow)
    p_nx = p_xyz.sum(axis=2)                   # p(next, xnow)
    p_x = p_nx.sum(axis=0, keepdims=True)      # p(xnow)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_full = p_xyz / p_ctx              # p(next | xnow, ynow)
        cond_own = np.broadcast_to((p_nx / p_x)[:, :, None], p_xyz.shape)
    mask = p_xyz > 0
    return float((p_xyz[mask] * np.log2(cond_full[mask] / cond_own[mask])).sum())


def ete_and_pvalue(sx: np.ndarray, sy: np.ndarray, rng: np.random.Generator,
                   with_pvalue: bool) -> tuple[float, float]:
    """ETE = TE - sesgo (media de B barajadas de la fuente). p-valor opcional
    contra la nula de R replicas con la fuente remuestreada por bloques."""
    te_obs = te_bits(sx, sy)
    bias = np.mean([te_bits(sx, rng.permutation(sy)) for _ in range(B_BIAS)])
    ete = te_obs - bias
    if not with_pvalue:
        return ete, np.nan
    n = len(sy)
    n_blocks = int(np.ceil(n / BLOCK))
    null = np.empty(R_NULL)
    for r in range(R_NULL):
        starts = rng.integers(0, n - BLOCK, size=n_blocks)
        fake = np.concatenate([sy[s:s + BLOCK] for s in starts])[:n]
        null[r] = te_bits(sx, fake)
    pval = float((np.sum(null >= te_obs) + 1) / (R_NULL + 1))
    return ete, pval


def bh_fdr(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg: devuelve mascara de significativos."""
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = alpha * (np.arange(1, m + 1)) / m
    passed = pvals[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    mask = np.zeros(m, dtype=bool)
    mask[order[:k]] = True
    return mask


def pair_sample(ret: pd.DataFrame, src: str, tgt: str,
                dates: pd.DatetimeIndex | None = None) -> tuple[np.ndarray, np.ndarray]:
    df = ret[[tgt, src]].dropna()
    if dates is not None:
        df = df.loc[df.index.isin(dates)]
    if len(df) < 30:
        return np.array([]), np.array([])
    return terciles(df[tgt].values), terciles(df[src].values)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    ret = load_returns()
    pairs = list(permutations(TICKERS, 2))
    crisis_dates = ret.index[np.zeros(len(ret), dtype=bool)]
    for a, b in CRISIS.values():
        crisis_dates = crisis_dates.union(ret.loc[a:b].index)
    calm_dates = ret.index.difference(crisis_dates)

    # ---------- Q1: periodo completo ----------
    print(f"Q1: {len(pairs)} pares, B_bias={B_BIAS}, R_null={R_NULL} ...")
    q1 = {}
    for src, tgt in pairs:
        sx, sy = pair_sample(ret, src, tgt)
        ete, pval = ete_and_pvalue(sx, sy, rng, with_pvalue=True)
        q1[(src, tgt)] = {"ete": ete, "pval": pval, "n": int(len(sx))}
    pvals = np.array([q1[p]["pval"] for p in pairs])
    signif = bh_fdr(pvals, ALPHA_FDR)
    for p, s in zip(pairs, signif):
        q1[p]["significant"] = bool(s)
    net = {t: sum(q1[(t, o)]["ete"] for o in TICKERS if o != t)
              - sum(q1[(o, t)]["ete"] for o in TICKERS if o != t) for t in TICKERS}
    print(f"  significativos post-FDR: {int(signif.sum())}/110")

    # ---------- Q2: crisis vs calma (terciles por muestra) ----------
    print("Q2: crisis vs calma ...")
    q2_pair = {}
    for src, tgt in pairs:
        sx_c, sy_c = pair_sample(ret, src, tgt, crisis_dates)
        sx_q, sy_q = pair_sample(ret, src, tgt, calm_dates)
        if len(sx_c) == 0 or len(sx_q) == 0:
            continue
        ete_c, _ = ete_and_pvalue(sx_c, sy_c, rng, with_pvalue=False)
        ete_q, _ = ete_and_pvalue(sx_q, sy_q, rng, with_pvalue=False)
        q2_pair[(src, tgt)] = {"crisis": ete_c, "calma": ete_q,
                               "n_crisis": int(len(sx_c)), "n_calma": int(len(sx_q))}
    diffs = np.array([v["crisis"] - v["calma"] for v in q2_pair.values()])
    etes_c = np.array([v["crisis"] for v in q2_pair.values()])
    etes_q = np.array([v["calma"] for v in q2_pair.values()])

    def ci_mean(x: np.ndarray) -> list[float]:
        reps = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(R_PAIRS)]
        return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]

    q2 = {
        "n_pairs": len(q2_pair),
        "n_days_crisis": int(len(crisis_dates)), "n_days_calma": int(len(calm_dates)),
        "mean_ete_crisis": float(etes_c.mean()), "ci_crisis": ci_mean(etes_c),
        "mean_ete_calma": float(etes_q.mean()), "ci_calma": ci_mean(etes_q),
        "mean_diff": float(diffs.mean()), "ci_diff": ci_mean(diffs),
        "ratio": float(etes_c.mean() / etes_q.mean()) if etes_q.mean() > 0 else None,
        "share_pairs_higher_in_crisis": float((diffs > 0).mean()),
    }
    # Desglose por episodio (solo descriptivo; episodios con muestra suficiente)
    episodes = {}
    for name, (a, b) in CRISIS.items():
        dates = ret.loc[a:b].index
        etes = []
        for src, tgt in pairs:
            sx, sy = pair_sample(ret, src, tgt, dates)
            if len(sx) < MIN_EPISODE_OBS:
                continue
            e, _ = ete_and_pvalue(sx, sy, rng, with_pvalue=False)
            etes.append(e)
        episodes[name] = ({"mean_ete": float(np.mean(etes)), "n_pairs": len(etes),
                           "n_days": int(len(dates))} if etes else
                          {"mean_ete": None, "n_pairs": 0, "n_days": int(len(dates)),
                           "excluded": f"muestra insuficiente (<{MIN_EPISODE_OBS} obs/par)"})
    print(f"  crisis={q2['mean_ete_crisis']:.5f} calma={q2['mean_ete_calma']:.5f} "
          f"diff={q2['mean_diff']:+.5f} CI={q2['ci_diff']}")

    # ---------- Q1b: vista anual descriptiva ----------
    print("Q1b: serie anual ...")
    annual = {}
    for year in range(ret.index[0].year + 1, ret.index[-1].year + 1):
        dates = ret.loc[str(year)].index
        etes = []
        for src, tgt in pairs:
            sx, sy = pair_sample(ret, src, tgt, dates)
            if len(sx) < MIN_YEAR_OBS:
                continue
            e, _ = ete_and_pvalue(sx, sy, rng, with_pvalue=False)
            etes.append(e)
        if etes:
            annual[year] = {"mean_ete": float(np.mean(etes)), "n_pairs": len(etes)}

    # ---------- Export ----------
    mat = pd.DataFrame(index=TICKERS, columns=TICKERS, dtype=float)
    pmat = pd.DataFrame(index=TICKERS, columns=TICKERS, dtype=float)
    smat = pd.DataFrame(False, index=TICKERS, columns=TICKERS)
    for (src, tgt), v in q1.items():
        mat.loc[src, tgt] = v["ete"]
        pmat.loc[src, tgt] = v["pval"]
        smat.loc[src, tgt] = v["significant"]
    mat.to_csv(OUT / "q1_ete_matrix.csv")
    pmat.to_csv(OUT / "q1_pvalues.csv")
    pd.Series(net).sort_values(ascending=False).to_csv(OUT / "q1_net_flow.csv",
                                                       header=["net_ete"])
    pd.DataFrame(annual).T.to_csv(OUT / "annual_ete.csv", index_label="year")

    results = {
        "config": {"states": 3, "tercile_scheme": "por muestra", "k": 1, "l": 1,
                   "b_bias": B_BIAS, "r_null": R_NULL, "block": BLOCK,
                   "alpha_fdr": ALPHA_FDR, "seed": SEED},
        "q1": {"pairs": {f"{s}->{t}": v for (s, t), v in q1.items()},
               "n_significant": int(signif.sum()), "net_flow": net},
        "q2": {**q2, "episodes": episodes},
        "annual": annual,
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
