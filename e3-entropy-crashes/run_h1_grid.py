"""Grid search de configs para H1: variar ventana de PE y cutoff de subset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from helpers import (
    CRASH_CUTOFF, MIN_EVENT_DAYS_MAIN, OUTPUTS,
    classify, compute_event_metrics, detect_events, load_price_ret_pe,
)

PE_WINDOWS = [75, 100, 120, 140, 160, 180]
SUBSET_UPPERS = [0.25, 0.30, 0.35, 0.40, None]
M = 3


def slope_pvalue(df):
    if len(df) < 4:
        return np.nan, np.nan
    X = sm.add_constant(df[["abs_drawdown"]])
    m = sm.OLS(df["delta_pe"], X).fit(cov_type="HC3")
    return float(m.params["abs_drawdown"]), float(m.pvalues["abs_drawdown"])


def main():
    rows = []
    for w in PE_WINDOWS:
        close, log_ret, pe = load_price_ret_pe(pe_window=w, m=M)
        events = detect_events(close)
        df_full = compute_event_metrics(events, log_ret, pe, pe_window=w,
                                          min_event_days=MIN_EVENT_DAYS_MAIN)
        df_full["type"] = classify(df_full["drawdown"], cutoff=CRASH_CUTOFF)

        df_full["abs_drawdown"] = df_full["drawdown"].abs()
        for upper in SUBSET_UPPERS:
            if upper is None:
                df = df_full.copy()
            else:
                df = df_full[df_full["abs_drawdown"] <= upper].copy()
            pull = df[df["type"] == "pullback"]
            crash = df[df["type"] == "crash"]
            b_all, p_all = slope_pvalue(df)
            b_pull, p_pull = slope_pvalue(pull)
            b_crash, p_crash = slope_pvalue(crash)
            X = sm.add_constant(df[["abs_drawdown"]])
            r2 = float(sm.OLS(df["delta_pe"], X).fit().rsquared) if len(df) >= 4 else np.nan
            rows.append({
                "pe_window": w,
                "subset_upper_pct": int(upper * 100) if upper is not None else 0,
                "n": len(df),
                "n_pullback": len(pull),
                "n_crash": len(crash),
                "beta_all": b_all,
                "p_all": p_all,
                "beta_pullback": b_pull,
                "p_pullback": p_pull,
                "beta_crash": b_crash,
                "p_crash": p_crash,
                "r2_all": r2,
            })
    grid = pd.DataFrame(rows)
    grid.to_csv(OUTPUTS / "e3_h1_grid.csv", index=False)
    print(f"Wrote {len(grid)} configs to e3_h1_grid.csv")

    best = grid.dropna(subset=["p_all"]).sort_values("p_all").head(5)
    print("\nTop 5 configs por p-value global:")
    print(best[["pe_window", "subset_upper_pct", "n", "beta_all",
                 "p_all", "r2_all"]].to_string(index=False))

    write_latex_table(grid)


def fmt_p(p):
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "$<$0{,}001"
    if p < 0.01:
        return f"{p:.3f}".replace(".", "{,}")
    return f"{p:.2f}".replace(".", "{,}")


def fmt_b(b):
    if pd.isna(b):
        return "--"
    return f"{b:+.3f}".replace(".", "{,}")


def write_latex_table(grid):
    grid = grid.copy()
    grid["cutoff_order"] = grid["subset_upper_pct"].replace(0, 999)
    grid = grid.sort_values(["cutoff_order", "pe_window"]).drop(columns=["cutoff_order"])
    lines = [
        r"\begin{longtable}{ccccccccc}",
        r"\toprule",
        r"Corte $|$dd$|$ & $W$ & $n$ & $\beta_{\text{todos}}$ & $p_{\text{todos}}$ & "
        r"$\beta_{\text{pull}}$ & $p_{\text{pull}}$ & $\beta_{\text{crash}}$ & $p_{\text{crash}}$ \\",
        r"\midrule",
    ]
    for _, r in grid.iterrows():
        corte = f"{int(r['subset_upper_pct'])}\\%" if r["subset_upper_pct"] > 0 else "sin corte"
        lines.append(
            f"{corte} & {int(r['pe_window'])} & {int(r['n'])} & "
            f"{fmt_b(r['beta_all'])} & {fmt_p(r['p_all'])} & "
            f"{fmt_b(r['beta_pullback'])} & {fmt_p(r['p_pullback'])} & "
            f"{fmt_b(r['beta_crash'])} & {fmt_p(r['p_crash'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\caption{Grid search H1: pendientes $\beta$ y $p$-values de OLS "
        r"$\Delta$PE $\sim$ drawdown para pullbacks (5-15\%), crashes ($\geq 15\%$) y todos, "
        r"variando ventana $W$ de PE (m=3, mín. 30 días) y corte superior de subset. "
        r"HC3 SE.} \label{tab:e3-h1-grid}",
        r"\end{longtable}",
    ]
    (OUTPUTS / "e3_h1_grid_table.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote LaTeX table to e3_h1_grid_table.tex")


if __name__ == "__main__":
    main()
