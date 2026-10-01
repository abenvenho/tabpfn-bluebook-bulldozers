"""Server-side cost quotes for every planned run — spends no quota, sees no result.

Usage: TABPFN_TOKEN=... python scripts/02_estimate_cost.py [--data-dir data/prepared]

Part 1 quotes the raw arm at each pre-registered context size; it is what froze
the PRIMARY context (PREREGISTRATION.md). Part 2 quotes the remaining blocks
(arms, context, modes, kaggle) so they can be scheduled inside the daily quota.
Each non-thinking run makes two predict calls (quantiles, then mean), so its
expected cost is about twice the single-call quote. Quotes use
tabpfn_client.estimate_cost(X_train, X_test=None, *, model_version, operation),
confirmed against tabpfn-client 0.6.1.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import to_matrix  # noqa: E402
from src.tabpfn_runner import build_arm, select_context  # noqa: E402


def quote(tabpfn_client, Xc, Xe=None, **kw):
    try:
        resp = tabpfn_client.estimate_cost(Xc, Xe, **kw)
        return int(resp.estimated_cost)
    except Exception as e:
        print(f"    ! {type(e).__name__}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/prepared")
    args = ap.parse_args()

    import tabpfn_client

    token = os.environ.get("TABPFN_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)
        print("[estimate] token taken from TABPFN_TOKEN")

    d = Path(args.data_dir)
    train = pd.read_parquet(d / "train.parquet")
    valid = pd.read_parquet(d / "valid.parquet")
    test = pd.read_parquet(d / "test.parquet")
    appendix = pd.read_parquet(d / "appendix.parquet")

    def X(df, arm="raw"):
        return to_matrix(build_arm(df, arm, appendix))[0]

    Xv_raw = X(valid)

    print("\n[estimate] Part 1 — raw arm, one predict call, validation split")
    for context in ("all", "200000", "100000", "50000"):
        ctx = select_context(train, context, "recent")
        c = quote(tabpfn_client, X(ctx), Xv_raw)
        print(f"  context {len(ctx):>7,} rows ({context:>6}): {c}")

    print("\n[estimate] Part 2 — remaining blocks (per call; x2 calls unless thinking)")
    plan = []
    for arm in ("clean", "appendix"):
        c = quote(tabpfn_client, X(train, arm), X(valid, arm))
        print(f"  arms     {arm:<9} all       : {c}")
        plan.append(("arms", c, 2))
    for n in ("200000", "100000", "50000"):
        for sampling in ("recent", "random"):
            ctx = select_context(train, n, sampling)
            c = quote(tabpfn_client, X(ctx), Xv_raw)
            print(f"  context  {sampling:<9} {n:>7}   : {c}")
            plan.append(("context", c, 2))
    c = quote(tabpfn_client, X(train), Xv_raw, model_version="v3.5-fast")
    print(f"  modes    fast      all       : {c}")
    plan.append(("modes", c, 2))
    ctx200 = select_context(train, "200000", "recent")
    cf = quote(tabpfn_client, X(ctx200), operation="thinking_fit")
    cp = quote(tabpfn_client, X(ctx200), Xv_raw, operation="thinking_predict")
    print(f"  modes    thinking  200000 fit: {cf}")
    print(f"  modes    thinking  200000 prd: {cp}")
    plan.append(("modes", cf, 1))
    plan.append(("modes", cp, 1))
    c = quote(tabpfn_client, X(train), X(test))
    print(f"  kaggle   raw       all (test): {c}")
    plan.append(("kaggle", c, 2))

    print("\n[estimate] expected spend per block (quote x calls):")
    totals = {}
    for block, c, k in plan:
        if c is None:
            totals[block] = None
        elif totals.get(block, 0) is not None:
            totals[block] = totals.get(block, 0) + c * k
    for block in ("arms", "context", "modes", "kaggle"):
        t = totals.get(block)
        print(f"  {block:<8}: {'unknown' if t is None else f'{t:,}'}")
    known = [t for t in totals.values() if t is not None]
    print(f"  total   : {sum(known):,}{'' if len(known) == len(totals) else ' (+ unknown)'}"
          f"  — daily quota 5,000,000")
    print("[estimate] done — nothing was spent.")


if __name__ == "__main__":
    main()
