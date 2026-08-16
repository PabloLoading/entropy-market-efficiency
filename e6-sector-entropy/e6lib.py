"""Maquinaria compartida de E6: usada por run_sectors.py (sectores SPDR) y
run_markets.py (indices nativos). Extraida sin cambios funcionales del
run_sectors.py original (spec e6-markets-analysis: refactor sin regresion)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from entropy import perm_entropy  # noqa: E402
from stats import midrank_percentiles  # noqa: E402

M, TAU = 3, 1


def rolling_pe(returns: pd.Series, w: int) -> pd.Series:
    vals = returns.values
    out = np.full(len(vals), np.nan)
    for i in range(w, len(vals) + 1):
        out[i - 1] = perm_entropy(vals[i - w:i], m=M, tau=TAU, normalize=True)
    return pd.Series(out, index=returns.index)


def block_bootstrap_ci(nov: pd.Series, reps: int, seed: int) -> tuple[float, float]:
    years = nov.index.year
    blocks = [nov.values[years == y] for y in np.unique(years)]
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    k = len(blocks)
    for r in range(reps):
        idx = rng.integers(0, k, size=k)
        means[r] = np.concatenate([blocks[i] for i in idx]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def annual_rankings(pe: pd.DataFrame, universe: list[str], rank_years,
                    out_path: Path) -> pd.DataFrame:
    annual = pe[universe].groupby(pe.index.year).mean()
    annual = annual.loc[annual.index.isin(rank_years)]
    # percentil mid-rank: (puesto - 0.5)/n, neutral exactamente 0.5 para todo n
    # (la formula previa puesto/n tenia neutral (n+1)/2n y sesgaba la
    # comparacion contra 0.5)
    pct = midrank_percentiles(annual)
    pct.to_csv(out_path, index_label="year")
    return pct


def analyze_rotation(pct: pd.DataFrame, names: dict[str, str]) -> dict:
    years = pct.index.tolist()
    spearmans = []
    for y0, y1 in zip(years[:-1], years[1:]):
        both = pct.loc[[y0, y1]].dropna(axis=1)
        if both.shape[1] >= 5:
            rho = stats.spearmanr(both.loc[y0], both.loc[y1]).statistic
            spearmans.append({"years": f"{y0}-{y1}", "rho": round(float(rho), 3)})
    rhos = [s["rho"] for s in spearmans]
    last = pct.idxmin(axis=1)
    repeats = (last.values[1:] == last.values[:-1]).mean()
    chance = float((1.0 / pct.notna().sum(axis=1)).mean())
    counts = last.value_counts().to_dict()
    return {
        "spearman_consecutivo_promedio": round(float(np.mean(rhos)), 3),
        "spearman_consecutivo_rango": [round(min(rhos), 3), round(max(rhos), 3)],
        "spearman_detalle": spearmans,
        "permanencia_ultimo_puesto": round(float(repeats), 3),
        "permanencia_azar": round(chance, 3),
        "ultimo_puesto_por_anio": {int(y): names[t] for y, t in last.items()},
        "veces_ultimo_puesto": {names[t]: int(c) for t, c in counts.items()},
    }


def analyze_percentiles(pct: pd.DataFrame, names: dict[str, str], seed: int,
                        reps: int = 4000) -> dict:
    """CI bootstrap del percentil promedio + masa direccional + etiquetas en
    niveles convencionales (90/95/99% bilateral) + binomial de ultimos puestos."""
    rng = np.random.default_rng(seed)
    last = pct.idxmin(axis=1)
    p_last = float((1.0 / pct.notna().sum(axis=1)).mean())
    res = {}
    for t in list(names):
        s = pct[t].dropna()
        ys = s.index.values
        means = np.array([s.loc[np.array(rng.choice(ys, size=len(ys)))].mean()
                          for _ in range(reps)])
        lo, hi = np.percentile(means, [2.5, 97.5])
        mass_above = float((means > 0.5).mean())
        conf = max(mass_above, 1 - mass_above)
        side = "eficiente" if mass_above >= 0.5 else "ineficiente"
        if conf >= 0.995:
            label = f"{side} (significativo al 99%)"
        elif conf >= 0.975:
            label = f"{side} (significativo al 95%)"
        elif conf >= 0.95:
            label = f"{side} (significativo al 90%)"
        else:
            label = "sin señal formal"
        n_last = int((last == t).sum())
        p_binom = float(1 - stats.binom.cdf(n_last - 1, len(s), p_last)) if n_last else 1.0
        res[t] = {
            "sector": names[t], "anios": int(len(s)),
            "percentil_promedio": round(float(s.mean()), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "confianza_direccional": round(conf, 3), "senal": label,
            "veces_ultimo": n_last, "p_binomial_ultimos": round(p_binom, 3),
        }
    return res


def analyze_crises(pe_short: pd.DataFrame, crises: dict, names: dict[str, str],
                   price_dir: Path, out_csv: Path, stats_json: Path,
                   reference: str | None = None, post_trough_days: int = 126,
                   pre_level_days: int = 63) -> pd.DataFrame:
    rows = []
    for name, (start, end) in crises.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        search = pe_short.loc[start:end + pd.Timedelta(days=post_trough_days * 1.5)]
        pre = pe_short.loc[:start].tail(pre_level_days)
        for t in pe_short.columns:
            if search[t].dropna().empty or pre[t].dropna().empty:
                continue
            s = search[t].dropna()
            px = pd.read_parquet(price_dir / f"{t}.parquet")["Adj Close"].dropna()
            px.index = pd.to_datetime(px.index).tz_localize(None)
            w = px.loc[start:end]
            dd = float((w / w.cummax() - 1).min()) if len(w) > 20 else np.nan
            rows.append({
                "episodio": name, "ticker": t,
                "sector": names.get(t, t),
                "pe_pre": round(float(pre[t].mean()), 5),
                "pe_min": round(float(s.min()), 5),
                "caida": round(float(pre[t].mean() - s.min()), 5),
                "fecha_min": s.idxmin().date().isoformat(),
                "drawdown_precio": round(dd, 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    sec = df[df["ticker"] != reference] if reference else df
    corr = {}
    for name, g in sec.groupby("episodio"):
        rho, p = stats.spearmanr(-g["drawdown_precio"], g["caida"])
        corr[name] = {"rho": round(float(rho), 3), "p": round(float(p), 3), "n": len(g)}
    rho, p = stats.spearmanr(-sec["drawdown_precio"], sec["caida"])
    corr["pooled"] = {"rho": round(float(rho), 3), "p": round(float(p), 4), "n": len(sec)}
    with open(stats_json, "w") as f:
        json.dump(corr, f, indent=2)
    return df


def coevolution(pe: pd.DataFrame, universe: list[str], out_csv: Path) -> pd.DataFrame:
    monthly = pe[universe].resample("ME").last()
    delta = monthly.diff().dropna(how="all")
    corr = delta.corr(min_periods=24).round(3)
    corr.to_csv(out_csv)
    return corr


def mann_kendall(series: pd.Series) -> dict:
    """Tendencia de una serie anual via Kendall tau contra el tiempo
    (equivalente asintotico del test de Mann-Kendall, como en 5.2)."""
    s = series.dropna()
    tau, p = stats.kendalltau(s.index.values.astype(float), s.values)
    if p < 0.05:
        direction = "creciente" if tau > 0 else "decreciente"
    else:
        direction = "sin tendencia"
    return {"tau": round(float(tau), 3), "p": round(float(p), 4),
            "tendencia": direction}
