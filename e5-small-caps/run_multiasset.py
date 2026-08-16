"""E5 fase 2: PE por accion-anio, S&P 600 vs S&P 500.

Produce en outputs/:
- stock_year_pe.parquet   tabla accion-anio (pe, ties, precio, dollar volume) por config
- multiasset_results.json medias/medianas por grupo con CI (bootstrap por accion),
                          brecha anual, Spearman PE vs dollar volume, ties por grupo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from entropy import perm_entropy  # noqa: E402

DATA = HERE / "data"
OUT_DIR = HERE / "outputs"
PRICES_600 = DATA / "prices_sp600"
PRICES_500 = HERE.parent / "e4-ml-crashes" / "data" / "prices"

YEARS = list(range(2015, 2026))
BIENNIA = [(y, y + 1) for y in range(2015, 2025, 2)]  # 2015-16 ... 2023-24
M = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # robustez D12: correr con 4
TAU = 1
MIN_DAYS = {"anual": 240, "bienal": 480}
MIN_PRICE = 5.0
BOOT_REPS = 2000
SEED = 42
SUFFIX = "" if M == 3 else f"_m{M}"
TIES_FILTER = 0.02  # mismo umbral del análisis de deciles m=3 (gen_charts_multiasset)


def ties_share(x: np.ndarray, m: int) -> float:
    """Fraction of embedding windows discarded because of repeated values."""
    n = len(x) - (m - 1)
    if n <= 0:
        return np.nan
    wins = np.lib.stride_tricks.sliding_window_view(x, m)
    unique_ok = np.array([np.unique(w).size == m for w in wins])
    return float(1.0 - unique_ok.mean())


def load_close(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty or "Adj Close" not in df.columns:
        return None
    out = df[["Adj Close", "Close", "Volume"]].dropna()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.loc["2015":"2025"]


def stock_rows(ticker: str, group: str, df: pd.DataFrame) -> list[dict]:
    ret = np.log(df["Adj Close"] / df["Adj Close"].shift(1)).dropna()
    rows = []
    periods = ([("anual", str(y), str(y)) for y in YEARS] +
               [("bienal", str(a), str(b)) for a, b in BIENNIA])
    for kind, y0, y1 in periods:
        r = ret.loc[y0:y1]
        if len(r) < MIN_DAYS[kind]:
            continue
        px = df["Close"].loc[y0:y1]
        vol = df["Volume"].loc[y0:y1]
        median_price = float(px.median())
        if median_price < MIN_PRICE:
            rows.append({"ticker": ticker, "group": group, "kind": kind,
                         "period": y0 if kind == "anual" else f"{y0}-{y1}",
                         "excluded": "price", "pe": np.nan})
            continue
        vals = r.values
        rows.append({
            "ticker": ticker, "group": group, "kind": kind,
            "period": y0 if kind == "anual" else f"{y0}-{y1}",
            "excluded": "",
            "pe": perm_entropy(vals, m=M, tau=TAU, normalize=True),
            "ties": ties_share(vals, m=M),
            "median_price": median_price,
            "dollar_volume": float((px * vol).median()),
            "n_days": len(r),
        })
    return rows


def bottom_decile_stats(sub: pd.DataFrame) -> dict:
    """Decile composition of per-stock mean PE (mirrors gen_charts_multiasset logic):
    share of small caps in bottom/top decile, with and without the ties filter,
    plus bottom-decile permanence between period halves."""
    out = {}
    for label, d in [("filtered", sub[sub["ticker"].isin(
            sub.groupby("ticker")["ties"].mean().loc[lambda s: s < TIES_FILTER].index)]),
            ("unfiltered", sub)]:
        ps = d.groupby(["ticker", "group"], as_index=False)["pe"].mean()
        if len(ps) < 50:
            continue
        dec = pd.qcut(ps["pe"], 10, labels=False, duplicates="drop")
        base = float((ps["group"] == "sp600").mean())
        out[label] = {
            "n_stocks": {k: int(v) for k, v in ps.groupby("group")["ticker"].count().items()},
            "base_share_small": round(base, 4),
            "bottom_decile_share_small": round(
                float((ps.loc[dec == 0, "group"] == "sp600").mean()), 4),
            "top_decile_share_small": round(
                float((ps.loc[dec == dec.max(), "group"] == "sp600").mean()), 4),
        }
    # Permanence of bottom-decile membership between halves (filtered universe).
    d = sub[sub["ticker"].isin(
        sub.groupby("ticker")["ties"].mean().loc[lambda s: s < TIES_FILTER].index)].copy()
    d["half"] = np.where(d["period"].str[:4].astype(int) <= 2019, "h1", "h2")
    halves = {}
    for h, dh in d.groupby("half"):
        ps = dh.groupby("ticker")["pe"].mean()
        if len(ps) < 50:
            continue
        halves[h] = set(ps[pd.qcut(ps, 10, labels=False, duplicates="drop") == 0].index)
    if len(halves) == 2:
        inter = halves["h1"] & halves["h2"]
        out["bottom_decile_permanence"] = {
            "n_h1": len(halves["h1"]), "n_h2": len(halves["h2"]),
            "n_both": len(inter),
            "share_of_h1": round(len(inter) / max(len(halves["h1"]), 1), 4),
        }
    return out


def boot_ci(per_stock: pd.DataFrame, stat: str, reps: int, seed: int) -> tuple[float, float]:
    """CI 95% of the group difference (small - large) of mean or median per-stock PE,
    resampling stocks with replacement within each group."""
    rng = np.random.default_rng(seed)
    small = per_stock.loc[per_stock["group"] == "sp600", "pe"].values
    large = per_stock.loc[per_stock["group"] == "sp500", "pe"].values
    fn = np.mean if stat == "mean" else np.median
    diffs = np.empty(reps)
    for r in range(reps):
        s = rng.choice(small, size=len(small), replace=True)
        l = rng.choice(large, size=len(large), replace=True)
        diffs[r] = fn(s) - fn(l)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    sp600_list = pd.read_csv(DATA / "sp600_tickers.csv")
    sp500_list = pd.read_csv(PRICES_500.parent / "sp500_tickers.csv")
    overlap = set(sp600_list["ticker_yahoo"]) & set(sp500_list["ticker_yahoo"])
    if overlap:
        print(f"Excluyendo del grupo small {len(overlap)} tickers presentes en ambas "
              f"listas: {sorted(overlap)}")

    rows: list[dict] = []
    missing = {"sp600": [], "sp500": []}
    for group, tickers, prices_dir in [
        ("sp600", [t for t in sp600_list["ticker_yahoo"] if t not in overlap], PRICES_600),
        ("sp500", sp500_list["ticker_yahoo"].tolist(), PRICES_500),
    ]:
        for t in tickers:
            df = load_close(prices_dir / f"{t}.parquet")
            if df is None or df.empty:
                missing[group].append(t)
                continue
            rows.extend(stock_rows(t, group, df))
        print(f"{group}: {len(tickers)} tickers, {len(missing[group])} sin datos")

    sy = pd.DataFrame(rows)
    sy.to_parquet(OUT_DIR / f"stock_year_pe{SUFFIX}.parquet")
    valid = sy[(sy["excluded"] == "") & sy["pe"].notna()].copy()

    results: dict = {"missing": {k: len(v) for k, v in missing.items()},
                     "overlap_excluded": len(overlap), "configs": {}}
    for kind, wlabel in [("anual", "w252"), ("bienal", "w504")]:
        sub = valid[valid["kind"] == kind]
        per_stock = sub.groupby(["ticker", "group"], as_index=False)["pe"].mean()
        n_excl_price = len(sy[(sy["kind"] == kind) & (sy["excluded"] == "price")])
        g = per_stock.groupby("group")["pe"]
        mean_s, mean_l = g.mean().get("sp600", np.nan), g.mean().get("sp500", np.nan)
        med_s, med_l = g.median().get("sp600", np.nan), g.median().get("sp500", np.nan)
        ci_mean = boot_ci(per_stock, "mean", BOOT_REPS, SEED)
        ci_med = boot_ci(per_stock, "median", BOOT_REPS, SEED)
        ties = sub.groupby("group")["ties"].mean()
        annual_gap = (sub.groupby(["period", "group"])["pe"].mean().unstack()
                      .assign(gap=lambda d: d["sp600"] - d["sp500"]))
        # Spearman PE vs log dollar volume per period, combined universe
        spearman = {}
        for p, dfp in sub.groupby("period"):
            dfp = dfp[dfp["dollar_volume"] > 0]
            x = np.log(dfp["dollar_volume"])
            spearman[p] = float(dfp["pe"].corr(x, method="spearman"))
        results["configs"][wlabel] = {
            "n_stocks": {k: int(v) for k, v in
                         per_stock.groupby("group")["ticker"].count().items()},
            "n_obs": int(len(sub)),
            "n_excluded_price": n_excl_price,
            "pe_mean": {"sp600": round(mean_s, 5), "sp500": round(mean_l, 5)},
            "pe_median": {"sp600": round(med_s, 5), "sp500": round(med_l, 5)},
            "diff_mean": round(mean_s - mean_l, 5),
            "diff_mean_ci95": [round(c, 5) for c in ci_mean],
            "diff_median": round(med_s - med_l, 5),
            "diff_median_ci95": [round(c, 5) for c in ci_med],
            "crosses_zero_mean": bool(ci_mean[0] <= 0 <= ci_mean[1]),
            "ties_share": {k: round(float(v), 4) for k, v in ties.items()},
            "annual_gap": {p: round(float(v), 5) for p, v in annual_gap["gap"].items()},
            "spearman_pe_dv": {p: round(v, 4) for p, v in spearman.items()},
            "bottom_decile": bottom_decile_stats(sub) if kind == "anual" else None,
        }
        r = results["configs"][wlabel]
        print(f"\n--- {wlabel} ({kind}) ---")
        print(f"  n stocks: {r['n_stocks']}  obs: {r['n_obs']}  "
              f"excluidas por precio: {r['n_excluded_price']}")
        print(f"  PE media  600={r['pe_mean']['sp600']:.4f}  500={r['pe_mean']['sp500']:.4f}  "
              f"diff={r['diff_mean']:+.4f}  CI={r['diff_mean_ci95']}")
        print(f"  PE mediana 600={r['pe_median']['sp600']:.4f}  500={r['pe_median']['sp500']:.4f}  "
              f"diff={r['diff_median']:+.4f}  CI={r['diff_median_ci95']}")
        print(f"  ties: {r['ties_share']}")
        sp_vals = list(r["spearman_pe_dv"].values())
        print(f"  Spearman PE~logDV: media {np.mean(sp_vals):+.3f} "
              f"(rango {min(sp_vals):+.3f} a {max(sp_vals):+.3f})")

    results["config"] = {"m": M, "tau": TAU, "boot_reps": BOOT_REPS, "seed": SEED}
    with open(OUT_DIR / f"multiasset_results{SUFFIX}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> {OUT_DIR / f'multiasset_results{SUFFIX}.json'}")


if __name__ == "__main__":
    main()
