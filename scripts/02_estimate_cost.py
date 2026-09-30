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

    cands = find_candidates(tabpfn_client)
    if not cands:
        print("[estimate] no cost/estimate callable found — paste this back:")
        print("  tabpfn_client:",
              [n for n in dir(tabpfn_client) if not n.startswith("_")])
        sys.exit(1)
    for k, (_, sig) in cands.items():
        print(f"[estimate] found {k}{sig}")

    for context in ("all", "200000", "100000", "50000"):
        ctx = select_context(train, context, "recent")
        Xc, _ = to_matrix(build_arm(ctx, "raw", None))
        yc = np.log1p(ctx[TARGET].to_numpy())
        got, errs = None, []
        for name, (fn, _) in cands.items():
            for call in (lambda f=fn: f(Xc, yc, Xe),
                         lambda f=fn: f(X_train=Xc, y_train=yc, X_test=Xe),
                         lambda f=fn: f(Xc, yc)):
                try:
                    got = (name, call())
                    break
                except TypeError as e:
                    errs.append(f"{name}: TypeError: {e}")
                except Exception as e:
                    errs.append(f"{name}: {type(e).__name__}: {e}")
                    break
            if got:
                break
        if got:
            print(f"[estimate] context {len(ctx):>7,} rows ({context}): {got[1]}"
                  f"  (via {got[0]})")
        else:
            print(f"[estimate] context {len(ctx):>7,} rows ({context}): "
                  f"no attempt succeeded:")
            for e in errs[:4]:
                print(f"    {e}")

    print("[estimate] done — nothing was spent. Paste this output back before "
          "any run (the PRIMARY context is frozen from it).")


if __name__ == "__main__":
    main()
