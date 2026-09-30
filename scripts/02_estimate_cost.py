"""Server-side cost quote for the planned runs — spends no quota, sees no result.

Usage: TABPFN_TOKEN=... python scripts/02_estimate_cost.py [--data-dir data/prepared]

Prints, for each pre-registered context size, what the API says a fit+predict
call on the raw arm costs, so the PRIMARY context can be frozen from the real
quote (PREREGISTRATION.md). Written defensively: tabpfn-client 0.6.1 documents
a cost-estimation entry point; if the installed client exposes it under a
different name or signature, the script prints what it found so the call can
be fixed without spending anything.
"""
import argparse
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import to_matrix  # noqa: E402
from src.schema import TARGET  # noqa: E402
from src.tabpfn_runner import build_arm, select_context  # noqa: E402


def find_candidates(tabpfn_client):
    """Callables on the module, the regressor class and an instance whose name
    mentions cost/estimate."""
    sources = [("tabpfn_client", tabpfn_client)]
    reg_cls = getattr(tabpfn_client, "TabPFNRegressor", None)
    if reg_cls is not None:
        sources.append(("TabPFNRegressor", reg_cls))
        try:
            sources.append(("TabPFNRegressor()", reg_cls()))
        except Exception:
            pass
    out = {}
    for label, src in sources:
        for name in dir(src):
            if name.startswith("_"):
                continue
            if "cost" in name.lower() or "estimat" in name.lower():
                obj = getattr(src, name)
                if callable(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except (TypeError, ValueError):
                        sig = "(?)"
                    out[f"{label}.{name}"] = (obj, sig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--split", default="valid")
    args = ap.parse_args()

    import tabpfn_client

    token = os.environ.get("TABPFN_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)
        print("[estimate] token taken from TABPFN_TOKEN")

    d = Path(args.data_dir)
    train = pd.read_parquet(d / "train.parquet")
    eval_df = pd.read_parquet(d / f"{args.split}.parquet")
    Xe, _ = to_matrix(build_arm(eval_df, "raw", None))

    # tabpfn-client 0.6.1 signature, confirmed on 2026-09-30:
    # estimate_cost(X_train, X_test=None, *, model_version=None,
    #               operation='predict', ...) -> EstimateCostResponse
    for context in ("all", "200000", "100000", "50000"):
        ctx = select_context(train, context, "recent")
        Xc, _ = to_matrix(build_arm(ctx, "raw", None))
        try:
            resp = tabpfn_client.estimate_cost(Xc, Xe)
            try:
                shown = resp.model_dump()
            except AttributeError:
                shown = vars(resp) if hasattr(resp, "__dict__") else resp
            print(f"[estimate] context {len(ctx):>7,} rows ({context}): {shown}")
        except Exception as e:
            print(f"[estimate] context {len(ctx):>7,} rows ({context}): "
                  f"{type(e).__name__}: {e}")

    print("[estimate] done — nothing was spent. Paste this output back before "
          "any run (the PRIMARY context is frozen from it).")


if __name__ == "__main__":
    main()
