"""Baselines: the competition's median benchmark and the engineered LightGBM.

Usage: python -m src.baselines [--data-dir data/prepared]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features import engineer_lgbm, to_matrix
from .metrics import rmsle
from .schema import DATE_PARSED, ID_COL, TARGET


def _save(tag: str, valid: pd.DataFrame, pred, extra: dict):
    Path("results/preds").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({ID_COL: valid[ID_COL], "y_true": valid[TARGET], "pred": pred})
    out.to_csv(f"results/preds/{tag}.csv", index=False)
    m = {"tag": tag, "split": "valid", "rmsle": rmsle(valid[TARGET], pred), **extra}
    with open(f"results/metrics/{tag}.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"[baselines] {tag}: RMSLE {m['rmsle']:.5f}")
    return m


def run_median(train: pd.DataFrame, valid: pd.DataFrame):
    med = float(train[TARGET].median())
    return _save("median_benchmark", valid, np.full(len(valid), med),
                 {"model": "global median", "median": med})


def run_lightgbm(train: pd.DataFrame, valid: pd.DataFrame, seed=42):
    import lightgbm as lgb

    t0 = time.time()
    tr = engineer_lgbm(train)
    va = engineer_lgbm(valid)

    # Internal early-stopping holdout: the last 10% of the training period.
    cut = train[DATE_PARSED].quantile(0.9)
    fit_mask = (train[DATE_PARSED] <= cut).to_numpy()

    X, cols = to_matrix(tr)
    Xv, _ = to_matrix(va, feature_cols=cols)
    for c in Xv.columns:  # align category dtypes with the training frame
        if str(X[c].dtype) == "category":
            Xv[c] = pd.Categorical(Xv[c].astype(object), categories=X[c].cat.categories)
    y = np.log1p(train[TARGET].to_numpy())

    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=5000, learning_rate=0.05,
        num_leaves=128, min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
        subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)
    try:  # LightGBM >= 4.7 sklearn API
        model.fit(X[fit_mask], y[fit_mask],
                  eval_X=[X[~fit_mask]], eval_y=[y[~fit_mask]],
                  eval_metric="rmse",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    except TypeError:
        model.fit(X[fit_mask], y[fit_mask],
                  eval_set=[(X[~fit_mask], y[~fit_mask])],
                  eval_metric="rmse",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    best_iter = model.best_iteration_ or model.n_estimators

    # Refit on the full training data with the selected number of trees.
    model_full = lgb.LGBMRegressor(
        objective="regression", n_estimators=best_iter, learning_rate=0.05,
        num_leaves=128, min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
        subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)
    model_full.fit(X, y)

    pred = np.expm1(model_full.predict(Xv))
    return _save("lgbm_engineered", valid, pred,
                 {"model": "LightGBM 2013-style engineered", "best_iteration": int(best_iter),
                  "n_features": len(cols), "wall_seconds": round(time.time() - t0, 1)})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/prepared")
    args = ap.parse_args(argv)
    d = Path(args.data_dir)
    train = pd.read_parquet(d / "train.parquet")
    valid = pd.read_parquet(d / "valid.parquet")
    run_median(train, valid)
    run_lightgbm(train, valid)


if __name__ == "__main__":
    main()
