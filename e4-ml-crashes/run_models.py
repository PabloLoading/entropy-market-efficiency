"""E4 tareas 2.1/2.2: walk-forward con RF/GB, OLS de referencia, tracking W&B.

Grid pre-declarado :
- Modelos: RandomForest, HistGradientBoosting. OLS referencia sin tuning.
- W de PE: 140, 180.
- Feature sets: baseline (5) vs con PE (6).
- 8 folds walk-forward: train expandiente desde 2000, val = ultimo 20%
  temporal del train, embargo 1 anio, test bloques de 2 anios (2009-2024).
- Hiperparametros por fold via val (grilla congelada). Test intocado.
- Sample weights 1/k por episodio (opcion --no-weights como sensibilidad).

W&B: 1 run por (modelo x W); folds logueados con step=fold; baseline y
con-PE dentro del mismo run. Artifacts: dataset (una vez) + modelos joblib.

Uso: run_models.py [--no-weights] [--no-wandb]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
PREDS = OUT / "preds"
MODELS = OUT / "models"
for d in (PREDS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(HERE / ".env")  # solo necesario para W&B (opcional)

SEED = 42
PE_WINDOWS = [140, 180]
BASE_FEATURES = ["drawdown_actual_t", "vol_realizada_t", "reversal_5d_t",
                  "market_dd_t", "rel_volume_t"]
TARGET = "trough_final"

RF_GRID = list(itertools.product([4, 6, 8], [5, 20]))       # (max_depth, min_samples_leaf)
GB_GRID = list(itertools.product([4, 6, 8], [5, 20]))       # idem, lr=0.1 fijo
TEST_BLOCKS = [(y, y + 1) for y in range(2009, 2025, 2)]     # 8 folds
VAL_FRACTION = 0.20
EMBARGO_YEARS = 1


def make_model(kind: str, params: tuple[int, int]):
    depth, leaf = params
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
            random_state=SEED, n_jobs=-1)
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, max_depth=depth,
        min_samples_leaf=leaf, random_state=SEED, early_stopping=False)


def weighted_mae(y, pred, w):
    return float(np.average(np.abs(y - pred), weights=w))


def r2(y, pred):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def tune_and_fit(kind, grid, train_sub, val, features, use_weights):
    w_tr = train_sub["weight"].values if use_weights else np.ones(len(train_sub))
    w_val = val["weight"].values if use_weights else np.ones(len(val))
    best, best_score = None, np.inf
    for params in grid:
        m = make_model(kind, params)
        m.fit(train_sub[features].values, train_sub[TARGET].values,
              sample_weight=w_tr)
        score = weighted_mae(val[TARGET].values,
                             m.predict(val[features].values), w_val)
        if score < best_score:
            best, best_score = params, score
    return best


def run_config(kind, grid, w_pe, df, use_weights, wandb_run, suffix="",
                variant="pe"):
    pe_col = f"{variant}_t_w{w_pe}"
    feats_pe = [pe_col] + BASE_FEATURES
    data = df.dropna(subset=feats_pe + [TARGET]).copy()

    fold_rows, pred_frames = [], []
    for fold, (y0, y1) in enumerate(TEST_BLOCKS, start=1):
        train_full = data[data["year"] <= y0 - 1 - EMBARGO_YEARS]
        test = data[data["year"].isin([y0, y1])]
        if len(train_full) < 100 or len(test) == 0:
            continue
        train_sorted = train_full.sort_values("entry_date")
        n_val = int(len(train_sorted) * VAL_FRACTION)
        train_sub, val = train_sorted.iloc[:-n_val], train_sorted.iloc[-n_val:]

        w_train = train_full["weight"].values if use_weights else np.ones(len(train_full))
        y_te = test[TARGET].values
        result = {"fold": fold, "test_years": f"{y0}-{y1}",
                  "n_train": len(train_full), "n_test": len(test)}
        preds = test[["ticker", "entry_date", "episode_id", "systemic",
                       TARGET]].copy()
        preds["fold"] = fold

        for label, feats in [("baseline", BASE_FEATURES), ("conpe", feats_pe)]:
            best = tune_and_fit(kind, grid, train_sub, val, feats, use_weights)
            m = make_model(kind, best)
            m.fit(train_full[feats].values, train_full[TARGET].values,
                  sample_weight=w_train)
            pred = m.predict(test[feats].values)
            result[f"mae_{label}"] = float(np.mean(np.abs(y_te - pred)))
            result[f"r2_{label}"] = r2(y_te, pred)
            result[f"params_{label}"] = str(best)
            preds[f"pred_{label}"] = pred
            joblib.dump(m, MODELS / f"{kind}_w{w_pe}{suffix}_f{fold}_{label}.joblib")

            # OLS referencia (sin tuning, mismos datos)
            ols = LinearRegression()
            ols.fit(train_full[feats].values, train_full[TARGET].values,
                    sample_weight=w_train)
            pred_ols = ols.predict(test[feats].values)
            result[f"mae_ols_{label}"] = float(np.mean(np.abs(y_te - pred_ols)))
            result[f"r2_ols_{label}"] = r2(y_te, pred_ols)
            preds[f"pred_ols_{label}"] = pred_ols

        result["delta_mae"] = result["mae_conpe"] - result["mae_baseline"]
        result["delta_r2"] = result["r2_conpe"] - result["r2_baseline"]
        fold_rows.append(result)
        pred_frames.append(preds)

        if wandb_run is not None:
            wandb_run.log({k: v for k, v in result.items()
                           if isinstance(v, (int, float))}, step=fold)

    results = pd.DataFrame(fold_rows)
    all_preds = pd.concat(pred_frames, ignore_index=True)
    return results, all_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--variant", choices=["pe", "wpe"], default="pe",
                    help="wpe: extension exploratoria, solo RF")
    args = ap.parse_args()
    use_weights = not args.no_weights
    variant = args.variant

    df = pd.read_parquet(DATA / "events_features.parquet")
    suffix = "" if use_weights else "_noweights"

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb

    dataset_logged = False
    summary_all = {}
    model_list = ([("rf", RF_GRID), ("gb", GB_GRID)] if variant == "pe"
                  else [("rf", RF_GRID)])
    for kind, grid in model_list:
        for w_pe in PE_WINDOWS:
            vtag = "" if variant == "pe" else f"_{variant}"
            config_name = f"{kind}{vtag}_w{w_pe}{suffix}"
            run = None
            if wandb is not None:
                tags = ["main"] if use_weights else ["no-weights"]
                if variant != "pe":
                    tags = [f"{variant}-extension"]
                run = wandb.init(
                    project="e4-pe-ml-crashes", name=config_name,
                    group=f"{kind}{vtag}_w{w_pe}",
                    tags=tags,
                    config={
                        "model": kind, "pe_window": w_pe, "variant": variant,
                        "features_baseline": BASE_FEATURES,
                        "grid": grid, "seed": SEED,
                        "sample_weights": use_weights,
                        "val_fraction": VAL_FRACTION,
                        "embargo_years": EMBARGO_YEARS,
                        "test_blocks": TEST_BLOCKS,
                    },
                    reinit=True)
                if not dataset_logged and use_weights:
                    art = wandb.Artifact("events-features", type="dataset")
                    art.add_file(str(DATA / "events_features.parquet"))
                    run.log_artifact(art)
                    dataset_logged = True

            print(f"\n=== {config_name} ===")
            results, preds = run_config(kind, grid, w_pe, df, use_weights, run,
                                         suffix=f"{vtag}{suffix}",
                                         variant=variant)
            print(results[["fold", "test_years", "n_train", "n_test",
                            "mae_baseline", "mae_conpe", "delta_mae",
                            "r2_baseline", "r2_conpe"]].to_string(index=False))

            agg = {
                "agg_mae_baseline": results["mae_baseline"].mean(),
                "agg_mae_conpe": results["mae_conpe"].mean(),
                "agg_delta_mae": results["delta_mae"].mean(),
                "agg_r2_baseline": results["r2_baseline"].mean(),
                "agg_r2_conpe": results["r2_conpe"].mean(),
                "agg_delta_r2": results["delta_r2"].mean(),
                "agg_mae_ols_baseline": results["mae_ols_baseline"].mean(),
                "agg_mae_ols_conpe": results["mae_ols_conpe"].mean(),
                "folds_pe_mejora": int((results["delta_mae"] < 0).sum()),
                "n_folds": len(results),
            }
            print({k: round(v, 5) if isinstance(v, float) else v
                   for k, v in agg.items()})
            summary_all[config_name] = agg

            results.to_csv(OUT / f"results_{config_name}.csv", index=False)
            preds.to_parquet(PREDS / f"preds_{config_name}.parquet", index=False)

            if run is not None:
                for k, v in agg.items():
                    run.summary[k] = v
                model_art = wandb.Artifact(f"models-{config_name}", type="model")
                for f in sorted(MODELS.glob(f"{kind}_w{w_pe}{vtag}{suffix}_f*.joblib")):
                    model_art.add_file(str(f))
                run.log_artifact(model_art)
                run.finish()

    with open(OUT / f"summary{suffix}.json", "w") as f:
        json.dump(summary_all, f, indent=2)
    print(f"\nGuardado summary{suffix}.json")


if __name__ == "__main__":
    main()
