"""Multiple-testing corrections and block-aware non-parametric tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm_bonferroni(p_values, alpha=0.05):
    """Holm-Bonferroni step-down correction.

    Args:
        p_values: array-like of p-values.
        alpha: family-wise error rate.

    Returns:
        DataFrame with p, p_adj, reject columns (preserves input order).
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adj_sorted = np.minimum.accumulate(
        np.maximum.accumulate(sorted_p * (n - np.arange(n))).clip(max=1.0)[::-1]
    )[::-1]
    adj = np.empty(n, dtype=float)
    adj[order] = adj_sorted
    return pd.DataFrame({"p": p, "p_adj": adj, "reject": adj < alpha})


def midrank_percentiles(df, axis=1):
    """Percentiles mid-rank de un ranking por fila: (puesto - 0.5) / n_validos.

    A diferencia de puesto/n (cuyo valor neutral es (n+1)/2n, no 0.5), el
    mid-rank promedia exactamente 0.5 para cualquier n, lo que habilita
    comparar percentiles promedio contra el neutral 0.5 incluso con paneles
    desbalanceados (n distinto por fila). Empates con rangos promedio; los
    NaN quedan fuera del ranking de su fila.

    Args:
        df: DataFrame de valores (filas = periodos, columnas = objetos).
        axis: eje del ranking (1 = rankea dentro de cada fila).

    Returns:
        DataFrame de percentiles en (0, 1), mismo shape que df.
    """
    ranks = df.rank(axis=axis, method="average")
    n_valid = df.notna().sum(axis=axis)
    return ranks.sub(0.5).div(n_valid, axis=1 - axis)


def kruskal_blocks(groups, block_size):
    """Kruskal-Wallis on one observation per block to mitigate autocorrelation.

    Args:
        groups: dict {group_name: array-like}.
        block_size: block length; takes every block_size-th observation.

    Returns:
        dict with statistic, p_value, n_per_group.
    """
    sampled = {g: np.asarray(v)[::block_size] for g, v in groups.items()}
    arrs = [sampled[g] for g in sampled]
    stat, p = stats.kruskal(*arrs)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_per_group": {g: len(sampled[g]) for g in sampled},
    }
