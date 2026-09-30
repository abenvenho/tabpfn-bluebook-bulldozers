"""Run one TabPFN arm (or its mock) and save predictions, quantiles and metrics.

Usage examples:
  python -m src.tabpfn_runner --split valid --arm raw --context all --mode base
  python -m src.tabpfn_runner --split valid --arm raw --context 50000 --sampling random
  python -m src.tabpfn_runner --split test  --arm raw --context all            # + submission
  MOCK: add --mock (used by the smoke test; no API, no token).

Target: log1p(SalePrice); predictions/quantiles back-transformed with expm1
(pre-registered). Quantiles q10..q90; the 80% interval is [q10, q90].
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features import clean_table, join_appendix, ordinal_encode, to_matrix
from .metrics import interval_coverage, rmsle
from .schema import DATE_COL, ID_COL, RAW_PREDICTORS, TARGET

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = [f"q{int(q * 100)}" for q in QUANTILES]

# Server-side model identifiers per mode. `base` uses the API default (TabPFN-3.5).
# scripts/00_check_api.py lists what the account actually exposes; override with --model-id.
MODE_MODEL_ID = {"base": None, "fast": "tabpfn-3.5-fast", "thinking": "tabpfn-3.5-thinking"}


def select_context(train: pd.DataFrame, context: str, sampling: str, seed=42):
    if context == "all":
        return train
    n = int(context)
    if n >= len(train):
        return train
    if sampling == "recent":
        return train.sort_values(DATE_COL).tail(n)
    return train.sample(n=n, random_state=seed)


def build_arm(df: pd.DataFrame, arm: str, appendix):
    """Return the frame handed to the predictor for a given arm (id/target removed later)."""
    if arm == "raw":
        out = df.copy()
    elif arm == "clean":
        out = clean_table(df)
    elif arm == "appendix":
        if appendix is None:
            raise FileNotFoundError("appendix arm requires Machine_Appendix.csv (see DATA_NOTICE.md)")
        out = join_appendix(df, appendix)
    else:
        raise ValueError(arm)
    if DATE_COL in out.columns:  # hand the date to TabPFN as an ISO string, not a datetime
        out[DATE_COL] = out[DATE_COL].dt.strftime("%Y-%m-%d")
    return out


def predict_mock(Xc, yc, Xe, seed=42):
    """Stand-in predictor for the smoke test: same output shape as the API path."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xc_e, Xe_e = ordinal_encode(Xc, Xe)
    n_hold = max(50, int(0.1 * len(Xc_e)))
    fit, hold = Xc_e.iloc[:-n_hold], Xc_e.iloc[-n_hold:]
    yfit, yhold = yc[:-n_hold], yc[-n_hold:]
    m = HistGradientBoostingRegressor(random_state=seed, max_iter=300)
    m.fit(fit, yfit)
    sigma = float(np.std(yhold - m.predict(hold))) or 0.3
    m.fit(Xc_e, yc)
    mean = m.predict(Xe_e)
    from scipy.stats import norm  # scipy ships with scikit-learn's dependency set
    qs = {q: mean + norm.ppf(q) * sigma for q in QUANTILES}
    return mean, qs, {"predictor": "mock (HistGradientBoosting + normal intervals)"}


def predict_api(Xc, yc, Xe, mode: str, model_id):
    """TabPFN via the Prior Labs API (tabpfn_client). Zero-shot, default settings."""
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor

    token = os.environ.get("TABPFN_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)

    kwargs = {}
    if model_id:
        try:
            reg = TabPFNRegressor(model_path=model_id)
            kwargs["model_path"] = model_id
        except TypeError:
            reg = TabPFNRegressor(model=model_id)
            kwargs["model"] = model_id
    else:
        reg = TabPFNRegressor()
    reg.fit(Xc, yc)

    try:
        qpred = reg.predict(Xe, output_type="quantiles", quantiles=QUANTILES)
        qs = {q: np.asarray(qpred[i]) for i, q in enumerate(QUANTILES)} \
            if isinstance(qpred, (list, tuple)) else {q: np.asarray(qpred[q]) for q in QUANTILES}
        mean = np.asarray(reg.predict(Xe))
    except Exception as e:  # quantile output unavailable on this mode/endpoint
        print(f"[tabpfn] quantile output unavailable ({e}); point predictions only")
        mean, qs = np.asarray(reg.predict(Xe)), {}
    return mean, qs, {"predictor": "tabpfn_client", "mode": mode, **kwargs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--arm", choices=["raw", "clean", "appendix"], default="raw")
    ap.add_argument("--context", default="all", help="'all' or an integer")
    ap.add_argument("--sampling", choices=["recent", "random"], default="recent")
    ap.add_argument("--mode", choices=["base", "fast", "thinking"], default="base")
    ap.add_argument("--model-id", default=None, help="override the server model identifier")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--data-dir", default="data/prepared")
    args = ap.parse_args(argv)

    d = Path(args.data_dir)
    train = pd.read_parquet(d / "train.parquet")
    eval_df = pd.read_parquet(d / f"{args.split}.parquet")
    appendix = pd.read_parquet(d / "appendix.parquet") if (d / "appendix.parquet").exists() else None

    tag = args.tag or "_".join(
        ["tabpfn", args.mode, args.arm, args.split, str(args.context), args.sampling]
        + (["mock"] if args.mock else []))

    ctx = select_context(train, args.context, args.sampling)
    Xc, cols = to_matrix(build_arm(ctx, args.arm, appendix))
    Xe, _ = to_matrix(build_arm(eval_df, args.arm, appendix), feature_cols=cols)
    yc = np.log1p(ctx[TARGET].to_numpy())

    t0 = time.time()
    if args.mock:
        mean_log, qs_log, info = predict_mock(Xc, yc, Xe)
    else:
        mean_log, qs_log, info = predict_api(Xc, yc, Xe, args.mode,
                                             args.model_id or MODE_MODEL_ID[args.mode])
    wall = round(time.time() - t0, 1)

    pred = np.expm1(mean_log)
    out = pd.DataFrame({ID_COL: eval_df[ID_COL].to_numpy(), "pred": pred})
    for q, col in zip(QUANTILES, QCOLS):
        if q in qs_log:
            out[col] = np.expm1(qs_log[q])

    Path("results/preds").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    m = {"tag": tag, "split": args.split, "arm": args.arm, "context": args.context,
         "sampling": args.sampling, "mode": args.mode, "n_context": int(len(ctx)),
         "n_eval": int(len(eval_df)), "wall_seconds": wall, **info}

    if args.split == "valid":
        out.insert(1, "y_true", eval_df[TARGET].to_numpy())
        m["rmsle"] = rmsle(out["y_true"], out["pred"])
        if "q10" in out.columns and "q90" in out.columns:
            m["coverage80"] = interval_coverage(out["y_true"], out["q10"], out["q90"])
        print(f"[tabpfn] {tag}: RMSLE {m['rmsle']:.5f}"
              + (f" | 80% coverage {m['coverage80']:.1%}" if "coverage80" in m else ""))
    else:
        sub = out[[ID_COL, "pred"]].rename(columns={"pred": TARGET})
        sub.to_csv(f"results/submission_{tag}.csv", index=False)
        print(f"[tabpfn] {tag}: submission -> results/submission_{tag}.csv "
              f"({len(sub)} rows). Upload it as a Kaggle late submission and transcribe "
              f"the score to results/kaggle_score.txt")

    out.to_csv(f"results/preds/{tag}.csv", index=False)
    with open(f"results/metrics/{tag}.json", "w") as f:
        json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
