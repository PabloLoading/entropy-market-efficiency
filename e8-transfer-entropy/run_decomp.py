"""E8 (extension): descomposicion del flujo en dos canales, direccion vs magnitud.

La ETE principal (run_te.py) usa 3 estados por terciles del retorno y mezcla
dependencia de signo y de magnitud. Aca se corre la misma maquinaria dos veces
por par, con estados binarios para que ambos canales sean comparables en bits:
- Canal direccion: estado = signo del retorno (r > 0), que descarta la magnitud.
- Canal magnitud:  estado = |r| por encima de su mediana, que descarta el signo.

Misma correccion effective (B_BIAS barajadas), misma nula por block bootstrap de
la fuente (R_NULL replicas, bloques de BLOCK dias) y mismo FDR al 5% por canal.
Se reporta por canal: ETE media, pares significativos, y la diferencia pareada
magnitud - direccion con CI por bootstrap sobre pares. Tambien la red de los
9 sectores con historia completa y el detalle del par XLF->XLK.

Output: outputs/decomp_results.json, outputs/decomp_pairs.csv
"""
from __future__ import annotations

import json
from itertools import permutations

import numpy as np
import pandas as pd

from run_te import (ALPHA_FDR, BLOCK, B_BIAS, OUT, R_NULL, R_PAIRS, SEED, TICKERS,
                    bh_fdr, load_returns)

FULL_HISTORY = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def te_bits_2(sx: np.ndarray, sy: np.ndarray) -> float:
    """TE(Y->X) plug-in en bits, k=l=1, sobre series de estados binarias."""
    idx = sx[1:] * 4 + sx[:-1] * 2 + sy[:-1]
    joint = np.bincount(idx, minlength=8).astype(float).reshape(2, 2, 2)
    n = joint.sum()
    p_xyz = joint / n
    p_ctx = p_xyz.sum(axis=0, keepdims=True)
    p_nx = p_xyz.sum(axis=2)
    p_x = p_nx.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_full = p_xyz / p_ctx
        cond_own = np.broadcast_to((p_nx / p_x)[:, :, None], p_xyz.shape)
    mask = p_xyz > 0
    return float((p_xyz[mask] * np.log2(cond_full[mask] / cond_own[mask])).sum())


def ete_pval_2(sx: np.ndarray, sy: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    te_obs = te_bits_2(sx, sy)
    bias = np.mean([te_bits_2(sx, rng.permutation(sy)) for _ in range(B_BIAS)])
    n = len(sy)
    n_blocks = int(np.ceil(n / BLOCK))
    null = np.empty(R_NULL)
    for r in range(R_NULL):
        starts = rng.integers(0, n - BLOCK, size=n_blocks)
        fake = np.concatenate([sy[s:s + BLOCK] for s in starts])[:n]
        null[r] = te_bits_2(sx, fake)
    pval = float((np.sum(null >= te_obs) + 1) / (R_NULL + 1))
    return te_obs - bias, pval


def states_sign(r: np.ndarray) -> np.ndarray:
    return (r > 0).astype(int)


def states_mag(r: np.ndarray) -> np.ndarray:
    a = np.abs(r)
    return (a > np.median(a)).astype(int)


def main() -> None:
    rng = np.random.default_rng(SEED)
    ret = load_returns()
    pairs = list(permutations(TICKERS, 2))
    rows = []
    print(f"decomp: {len(pairs)} pares x 2 canales, B_bias={B_BIAS}, R_null={R_NULL} ...")
    for i, (src, tgt) in enumerate(pairs):
        df = ret[[tgt, src]].dropna()
        r_t, r_s = df[tgt].values, df[src].values
        ete_d, p_d = ete_pval_2(states_sign(r_t), states_sign(r_s), rng)
        ete_m, p_m = ete_pval_2(states_mag(r_t), states_mag(r_s), rng)
        rows.append({"src": src, "tgt": tgt, "n": len(df),
                     "ete_dir": ete_d, "p_dir": p_d, "ete_mag": ete_m, "p_mag": p_m})
        if (i + 1) % 22 == 0:
            print(f"  {i + 1}/{len(pairs)}")
    tab = pd.DataFrame(rows)
    tab["sig_dir"] = bh_fdr(tab["p_dir"].values, ALPHA_FDR)
    tab["sig_mag"] = bh_fdr(tab["p_mag"].values, ALPHA_FDR)
    tab["diff"] = tab["ete_mag"] - tab["ete_dir"]

    def ci_mean(x: np.ndarray) -> list[float]:
        reps = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(R_PAIRS)]
        return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]

    def summarize(t: pd.DataFrame) -> dict:
        return {
            "n_pairs": int(len(t)),
            "dir": {"mean_ete": float(t.ete_dir.mean()), "ci_mean": ci_mean(t.ete_dir.values),
                    "n_signif": int(t.sig_dir.sum()), "share_signif": float(t.sig_dir.mean())},
            "mag": {"mean_ete": float(t.ete_mag.mean()), "ci_mean": ci_mean(t.ete_mag.values),
                    "n_signif": int(t.sig_mag.sum()), "share_signif": float(t.sig_mag.mean())},
            "diff_mag_minus_dir": {"mean": float(t["diff"].mean()), "ci": ci_mean(t["diff"].values),
                                   "share_pairs_mag_gt_dir": float((t["diff"] > 0).mean())},
        }

    full = tab[tab.src.isin(FULL_HISTORY) & tab.tgt.isin(FULL_HISTORY)].copy()
    # FDR recomputado dentro de la red de 9
    full["sig_dir"] = bh_fdr(full["p_dir"].values, ALPHA_FDR)
    full["sig_mag"] = bh_fdr(full["p_mag"].values, ALPHA_FDR)

    # detalle XLF -> XLK (probabilidades condicionales, mismo criterio de terciles de run_te)
    df = ret[["XLK", "XLF"]].dropna()
    from run_te import terciles
    s_t, s_s = terciles(df["XLK"].values), terciles(df["XLF"].values)
    ext_s, ext_t = s_s[:-1] != 1, s_t[1:] != 1
    up, dn = s_s[:-1] == 2, s_s[:-1] == 0
    xlf_xlk = {
        "n": int(len(ext_t)),
        "p_tec_extremo": float(ext_t.mean()),
        "p_tec_extremo_dado_fin_extremo": float(ext_t[ext_s].mean()),
        "p_tec_extremo_dado_fin_normal": float(ext_t[~ext_s].mean()),
        "p_tec_sube_dado_fin_sube": float((s_t[1:][up] == 2).mean()),
        "p_tec_baja_dado_fin_sube": float((s_t[1:][up] == 0).mean()),
        "p_tec_sube_dado_fin_baja": float((s_t[1:][dn] == 2).mean()),
        "p_tec_baja_dado_fin_baja": float((s_t[1:][dn] == 0).mean()),
        "ete_dir": float(tab.loc[(tab.src == "XLF") & (tab.tgt == "XLK"), "ete_dir"].iloc[0]),
        "ete_mag": float(tab.loc[(tab.src == "XLF") & (tab.tgt == "XLK"), "ete_mag"].iloc[0]),
    }

    # ranking de flujo neto por canal (red completa)
    def net(t: pd.DataFrame, col: str) -> dict:
        return {k: float(t[t.src == k][col].sum() - t[t.tgt == k][col].sum()) for k in TICKERS}

    results = {
        "config": {"states": 2, "dir_state": "r > 0", "mag_state": "|r| > mediana",
                   "k": 1, "l": 1, "b_bias": B_BIAS, "r_null": R_NULL, "block": BLOCK,
                   "alpha_fdr": ALPHA_FDR, "seed": SEED},
        "all_110": summarize(tab),
        "full_history_72": summarize(full),
        "net_flow_dir": net(tab, "ete_dir"),
        "net_flow_mag": net(tab, "ete_mag"),
        "xlf_to_xlk": xlf_xlk,
    }
    tab.to_csv(OUT / "decomp_pairs.csv", index=False)
    with open(OUT / "decomp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    a = results["all_110"]
    print(f"110 pares | dir: media {a['dir']['mean_ete'] * 1000:.2f} mils, signif {a['dir']['n_signif']}"
          f" | mag: media {a['mag']['mean_ete'] * 1000:.2f} mils, signif {a['mag']['n_signif']}"
          f" | diff {a['diff_mag_minus_dir']['mean'] * 1000:+.2f} mils CI {[round(x * 1000, 2) for x in a['diff_mag_minus_dir']['ci']]}")
    print(f"-> {OUT / 'decomp_results.json'}")


if __name__ == "__main__":
    main()
