"""E7: PE intra-franja agrupada sobre SPY.

Configs (D3): horaria m=3 y m=4 (5-min), media hora m=3 (5-min), curva de 5
minutos m=3 y m=4 (1-min). Sin rolling. Bootstrap por dias (D4) con muestra de
dias compartida entre franjas (preserva el pareo para diferencias); robustez
por bloques mensuales. Comparaciones pre-declaradas: apertura (9:30-10:30) vs
mediodia (12:30-13:30) vs cierre (15:30-16:00); en las configs de curva se
evaluan agregando los grupos de 5 min internos a cada ventana.

Outputs: intraday_results.json, curve.csv, evolution.csv, evolution_gap.json
"""
from __future__ import annotations

import datetime as dt
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
REPS = 2000
SEED = 42

PATTERNS = {m: {p: i for i, p in enumerate(permutations(range(m)))} for m in (3, 4)}


def load(freq: str) -> pd.DataFrame:
    df = pd.read_parquet(HERE / "data" / f"spy_{freq}_regular.parquet").reset_index()
    df["day"] = df["date"].dt.date
    return df


def day_pattern_counts(df: pd.DataFrame, edges: list[dt.time], step_min: int,
                       m: int) -> tuple[np.ndarray, list, np.ndarray, int]:
    """Matriz (dias x franjas x m!) de conteos de patrones intra-franja.
    Devuelve (counts, labels, days, ties_descartados)."""
    npat = len(PATTERNS[m])
    days = np.array(sorted(df["day"].unique()))
    day_ix = {d: i for i, d in enumerate(days)}
    nfr = len(edges) - 1
    counts = np.zeros((len(days), nfr, npat), dtype=np.int32)
    ties = 0

    close = df["close"].values
    ret = np.full(len(df), np.nan)
    ret[1:] = np.log(close[1:] / close[:-1])
    # invalidar retornos que cruzan dias o huecos de tiempo
    tsec = df["date"].values.astype("datetime64[s]").astype(np.int64)
    same_day = df["day"].values[1:] == df["day"].values[:-1]
    consec = (tsec[1:] - tsec[:-1]) == step_min * 60
    ret[1:][~(same_day & consec)] = np.nan

    times = df["date"].dt.time.values
    fr_ix = np.full(len(df), -1)
    for i in range(nfr):
        sel = (times >= edges[i]) & (times < edges[i + 1])
        fr_ix[sel] = i
    # un retorno pertenece a la franja de su barra final; ademas exigir que
    # TODO el patron caiga en la misma franja: invalidar retornos cuya barra
    # inicial este en otra franja
    prev_fr = np.roll(fr_ix, 1)
    ret[(fr_ix != prev_fr)] = np.nan

    d_ix = np.array([day_ix[d] for d in df["day"].values])
    # ventanas deslizantes de m retornos consecutivos validos y misma franja
    from numpy.lib.stride_tricks import sliding_window_view
    if len(ret) < m:
        return counts, [], days, 0
    W = sliding_window_view(ret, m)
    F = sliding_window_view(fr_ix, m)
    D = sliding_window_view(d_ix, m)
    valid = ~np.isnan(W).any(axis=1)
    valid &= (F == F[:, [0]]).all(axis=1) & (D == D[:, [0]]).all(axis=1) & (F[:, 0] >= 0)
    Wv, fv, dv = W[valid], F[valid][:, 0], D[valid][:, 0]
    # descartar ties (valores repetidos), consistente con common.perm_entropy
    order = np.argsort(Wv, axis=1, kind="stable")
    sorted_w = np.take_along_axis(Wv, order, axis=1)
    no_tie = (np.diff(sorted_w, axis=1) != 0).all(axis=1)
    ties = int((~no_tie).sum())
    Wv, fv, dv = Wv[no_tie], fv[no_tie], dv[no_tie]
    codes = np.zeros(len(Wv), dtype=np.int64)
    ranks = np.argsort(np.argsort(Wv, axis=1, kind="stable"), axis=1)
    mult = np.array([ (m - 1 - k) and 0 or 0 for k in range(m)])  # placeholder
    # codificar permutacion via lookup
    perm_list = list(PATTERNS[m])
    lut = {p: i for i, p in enumerate(perm_list)}
    codes = np.array([lut[tuple(r)] for r in ranks], dtype=np.int32)
    np.add.at(counts, (dv, fv, codes), 1)
    labels = [f"{edges[i].strftime('%H:%M')}" for i in range(nfr)]
    return counts, labels, days, ties


def entropy(freqs: np.ndarray) -> float | np.ndarray:
    tot = freqs.sum(axis=-1, keepdims=True)
    p = np.where(tot > 0, freqs / np.maximum(tot, 1), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0).sum(axis=-1)
    return h / np.log2(freqs.shape[-1])


def bootstrap_weights(n_days: int, groups: np.ndarray, reps: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Matriz (reps x dias) de pesos: cuantas veces entro cada dia al resample.
    groups: id de bloque por dia (dias: identidad; meses: id de mes)."""
    uniq = np.unique(groups)
    W = np.zeros((reps, n_days), dtype=np.float64)
    for r in range(reps):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        cnt = pd.Series(samp).value_counts()
        w = np.zeros(n_days)
        for g, c in cnt.items():
            w[groups == g] = c
        W[r] = w
    return W


def analyze(counts: np.ndarray, labels: list, days: np.ndarray,
            W_days: np.ndarray, W_months: np.ndarray) -> dict:
    total = counts.sum(axis=0)  # franjas x npat
    pe_full = entropy(total)
    # bootstrap: (reps x dias) @ (dias x franjas*npat)
    flat = counts.reshape(len(days), -1).astype(np.float64)
    boot = (W_days @ flat).reshape(len(W_days), *total.shape)
    pe_boot = entropy(boot)  # reps x franjas
    lo, hi = np.percentile(pe_boot, [2.5, 97.5], axis=0)
    res = {"franjas": [
        {"label": labels[i], "pe": round(float(pe_full[i]), 5),
         "ci95": [round(float(lo[i]), 5), round(float(hi[i]), 5)],
         "n_patrones": int(total[i].sum())}
        for i in range(len(labels))]}
    res["_pe_boot"] = pe_boot
    res["_pe_boot_months"] = entropy((W_months @ flat).reshape(len(W_months), *total.shape))
    return res


def comparisons(res: dict, labels: list, pairs: list[tuple[str, str]]) -> dict:
    out = {}
    ix = {l: i for i, l in enumerate(labels)}
    pe = {f["label"]: f["pe"] for f in res["franjas"]}
    for a, b in pairs:
        d = pe[a] - pe[b]
        for key, boot in [("dias", res["_pe_boot"]), ("meses", res["_pe_boot_months"])]:
            db = boot[:, ix[a]] - boot[:, ix[b]]
            lo, hi = np.percentile(db, [2.5, 97.5])
            out.setdefault(f"{a} vs {b}", {"diff": round(float(d), 5)})[f"ci95_{key}"] = \
                [round(float(lo), 5), round(float(hi), 5)]
    return out


WINDOWS = {"apertura": ("09:30", "10:30"), "mediodia": ("12:30", "13:30"),
           "cierre": ("15:30", "16:00")}
WINDOW_PAIRS = [("apertura", "mediodia"), ("apertura", "cierre"),
                ("cierre", "mediodia")]


def window_comparisons(counts: np.ndarray, labels: list,
                       W_days: np.ndarray, W_months: np.ndarray) -> dict:
    """Comparaciones pre-declaradas sobre ventanas de tiempo, agregando las
    franjas internas a cada ventana (permite correrlas en la curva de 5 min).
    Los patrones siguen siendo intra-franja: la agregacion suma conteos, no
    cruza fronteras."""
    t = [dt.time.fromisoformat(l) for l in labels]
    names = list(WINDOWS)
    sel = [[i for i, ti in enumerate(t)
            if dt.time.fromisoformat(a) <= ti < dt.time.fromisoformat(b)]
           for a, b in WINDOWS.values()]
    agg = np.stack([counts[:, ix, :].sum(axis=1) for ix in sel], axis=1)
    total = agg.sum(axis=0)  # ventanas x npat
    pe_full = entropy(total)
    flat = agg.reshape(len(agg), -1).astype(np.float64)
    out: dict = {}
    for key, W in [("dias", W_days), ("meses", W_months)]:
        pb = entropy((W @ flat).reshape(len(W), *total.shape))
        for a, b in WINDOW_PAIRS:
            ia, ib = names.index(a), names.index(b)
            db = pb[:, ia] - pb[:, ib]
            lo, hi = np.percentile(db, [2.5, 97.5])
            out.setdefault(f"{a} vs {b}",
                           {"diff": round(float(pe_full[ia] - pe_full[ib]), 5)})[
                f"ci95_{key}"] = [round(float(lo), 5), round(float(hi), 5)]
    return out


def ordinal_percentiles(counts: np.ndarray, labels: list,
                        rng: np.random.Generator, reps: int = 2000) -> list[dict]:
    """Consistencia ordinal : ranking intra-dia de franjas por su PE
    diaria; percentil promedio por franja con CI (bootstrap de dias) vs 0,5.
    Solo compara franjas con la MISMA cantidad tipica de patrones por dia: la
    entropia estimada con menos muestras esta sesgada hacia abajo y una franja
    mas corta rankearia ultima por mecanica, no por estructura."""
    tot = counts.sum(axis=2)  # dias x franjas
    typical = np.median(tot, axis=0)
    mode_size = np.median(typical)
    eq = np.where(typical == mode_size)[0]
    daily_pe = entropy(counts.astype(np.float64))
    daily_pe = np.where(tot > 0, daily_pe, np.nan)
    df = pd.DataFrame(daily_pe[:, eq], columns=[labels[i] for i in eq])
    ranks = df.rank(axis=1, method="average")
    # percentil mid-rank: (puesto - 0.5)/n -> neutral exactamente 0.5 para todo n
    nvalid = ranks.notna().sum(axis=1)
    pct = (ranks.sub(0.5)).div(nvalid, axis=0)
    means = pct.mean().values
    n = len(pct)
    idx = rng.integers(0, n, size=(reps, n))
    boot = np.stack([np.nanmean(pct.values[idx[r]], axis=0) for r in range(reps)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    excluded = [labels[i] for i in range(len(labels)) if i not in eq]
    out = [{"label": df.columns[i], "percentil_promedio": round(float(means[i]), 4),
            "ci95": [round(float(lo[i]), 4), round(float(hi[i]), 4)]}
           for i in range(len(df.columns))]
    if excluded:
        out.append({"excluidas_por_tamano": excluded})
    return out


def time_edges(start: str, end: str, minutes: int) -> list[dt.time]:
    t0 = dt.datetime.combine(dt.date(2000, 1, 1), dt.time.fromisoformat(start))
    t1 = dt.datetime.combine(dt.date(2000, 1, 1), dt.time.fromisoformat(end))
    edges = []
    while t0 < t1:
        edges.append(t0.time())
        t0 += dt.timedelta(minutes=minutes)
    edges.append(t1.time())  # borde final (la ultima franja puede ser mas corta)
    return edges


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    d5, d1 = load("5min"), load("1min")

    results: dict = {}
    # pesos bootstrap compartidos por dataset (mismos dias)
    for name, df, step, edges, m in [
        ("horaria_m3", d5, 5, time_edges("09:30", "16:00", 60), 3),
        ("horaria_m4", d5, 5, time_edges("09:30", "16:00", 60), 4),
        ("media_hora_m3", d5, 5, time_edges("09:30", "16:00", 30), 3),
        ("curva_5min_m3", d1, 1, time_edges("09:30", "16:00", 5), 3),
        # agregada pre-declarada: sensibilidad m=4 en la escala fina;
        # va ultima para no alterar la secuencia del RNG de las configs previas
        ("curva_5min_m4", d1, 1, time_edges("09:30", "16:00", 5), 4),
    ]:
        counts, labels, days, ties = day_pattern_counts(df, edges, step, m)
        months = np.array([pd.Timestamp(d).strftime("%Y-%m") for d in days])
        month_ids = pd.factorize(months)[0]
        Wd = bootstrap_weights(len(days), np.arange(len(days)), REPS, rng)
        Wm = bootstrap_weights(len(days), month_ids, REPS, rng)
        res = analyze(counts, labels, days, Wd, Wm)
        res["ties_descartados"] = ties
        print(f"{name}: {len(labels)} franjas, {int(counts.sum()):,} patrones, ties={ties}")
        pes = [f["pe"] for f in res["franjas"]]
        print(f"  PE min={min(pes):.4f} ({res['franjas'][int(np.argmin(pes))]['label']}) "
              f"max={max(pes):.4f} ({res['franjas'][int(np.argmax(pes))]['label']})")
        if name == "horaria_m3":
            # evolucion anual (2009-2020) sobre la config horaria
            years = np.array([pd.Timestamp(d).year for d in days])
            rows = []
            gaps = {}
            for y in range(2009, 2021):
                sel = years == y
                cy = counts[sel]
                tot = cy.sum(axis=0)
                pe_y = entropy(tot)
                rows.append({"year": y, **{labels[i]: round(float(pe_y[i]), 5)
                                            for i in range(len(labels))}})
                # gap apertura-mediodia con CI por bootstrap de dias del año
                flat = cy.reshape(sel.sum(), -1).astype(np.float64)
                Wy = bootstrap_weights(sel.sum(), np.arange(sel.sum()), 1000, rng)
                pb = entropy((Wy @ flat).reshape(1000, *tot.shape))
                g = pb[:, 0] - pb[:, 3]  # 9:30 franja vs 12:30 franja
                gaps[y] = {"gap": round(float(pe_y[0] - pe_y[3]), 5),
                           "ci95": [round(float(np.percentile(g, 2.5)), 5),
                                     round(float(np.percentile(g, 97.5)), 5)]}
            pd.DataFrame(rows).to_csv(OUT / "evolution.csv", index=False)
            with open(OUT / "evolution_gap.json", "w") as f:
                json.dump(gaps, f, indent=2)
        # consistencia ordinal (D4b); en curva m=4 hay 1 patron/dia y la PE
        # diaria por grupo es degenerada, se omite
        if name != "curva_5min_m4":
            res["percentiles"] = ordinal_percentiles(counts, labels, rng)
            ext = [p for p in res["percentiles"]
                   if p.get("label") in ("09:30", "15:30", "15:55", "12:30")]
            for p in ext:
                print(f"    percentil {p['label']}: {p['percentil_promedio']:.3f} CI={p['ci95']}")
        if name == "horaria_m4":
            # gap cierre-mediodia por año (hallazgo principal, reproducible)
            years4 = np.array([pd.Timestamp(d).year for d in days])
            i_cl, i_md = labels.index("15:30"), labels.index("12:30")
            gaps4 = {}
            for y in range(2009, 2021):
                sel = years4 == y
                cy = counts[sel]
                pe_y = entropy(cy.sum(axis=0))
                flat = cy.reshape(sel.sum(), -1).astype(np.float64)
                Wy = bootstrap_weights(sel.sum(), np.arange(sel.sum()), 1000, rng)
                pb = entropy((Wy @ flat).reshape(1000, len(labels), -1))
                g = pb[:, i_cl] - pb[:, i_md]
                gaps4[y] = {"gap": round(float(pe_y[i_cl] - pe_y[i_md]), 5),
                            "ci95": [round(float(np.percentile(g, 2.5)), 5),
                                      round(float(np.percentile(g, 97.5)), 5)]}
            with open(OUT / "evolution_close_gap_m4.json", "w") as f:
                json.dump(gaps4, f, indent=2)
        if name.startswith("curva_5min"):
            fname = "curve.csv" if name.endswith("m3") else "curve_m4.csv"
            pd.DataFrame([{"time": f["label"], "pe": f["pe"], "lo": f["ci95"][0],
                           "hi": f["ci95"][1], "n": f["n_patrones"]}
                          for f in res["franjas"]]).to_csv(OUT / fname, index=False)
        del res["_pe_boot_months"]
        # comparaciones pre-declaradas: por franja exacta en las horarias, por
        # ventanas agregadas en las curvas
        if name in ("horaria_m3", "horaria_m4"):
            res["comparaciones"] = comparisons(
                {**res, "_pe_boot_months": entropy((Wm @ counts.reshape(len(days), -1)
                    .astype(np.float64)).reshape(REPS, len(labels), -1))},
                labels, [("09:30", "12:30"), ("09:30", "15:30"), ("15:30", "12:30")])
        elif name.startswith("curva_5min"):
            res["comparaciones"] = window_comparisons(counts, labels, Wd, Wm)
        del res["_pe_boot"]
        results[name] = res

    with open(OUT / "intraday_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n-> {OUT / 'intraday_results.json'}")


if __name__ == "__main__":
    main()
