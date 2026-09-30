"""Check the Prior Labs API before spending the quota: token, tiny fit/predict,
quantile output, and a cost quote for the planned context sizes.

Usage: TABPFN_TOKEN=... python scripts/00_check_api.py
"""
import os
import sys

import numpy as np
import pandas as pd


def fail(msg):
    print(f"[check_api] FAIL: {msg}")
    sys.exit(1)


def main():
    token = os.environ.get("TABPFN_TOKEN")
    try:
        import tabpfn_client
        from tabpfn_client import TabPFNRegressor
    except ImportError:
        fail("tabpfn-client is not installed (pip install -r requirements.txt)")

    if token:
        tabpfn_client.set_access_token(token)
        print("[check_api] token taken from TABPFN_TOKEN")
    else:
        print("[check_api] TABPFN_TOKEN not set — tabpfn_client will use its own "
              "login flow / cached token if available")

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.choice(list("xyz"), 60),
                      "c": rng.integers(0, 5, 60)})
    y = X["a"] * 2 + X["c"] + rng.normal(scale=0.1, size=60)

    try:
        reg = TabPFNRegressor()
        reg.fit(X.iloc[:50], y[:50])
        pred = reg.predict(X.iloc[50:])
        print(f"[check_api] point predictions OK ({len(pred)} rows, default model = "
              f"server default, expected TabPFN-3.5)")
    except Exception as e:
        msg = str(e)
        if any(k in msg.lower() for k in ("connect", "network", "resolve", "timeout",
                                          "403", "proxy")):
            fail(f"could not reach the API ({e}).\n"
                 "  The sandbox egress allowlist may be blocking priorlabs.ai — either\n"
                 "  allow it in the Claude network settings or run this on a machine\n"
                 "  with open network access.")
        fail(f"API call failed: {e}")

    try:
        q = reg.predict(X.iloc[50:], output_type="quantiles",
                        quantiles=[0.1, 0.5, 0.9])
        n = len(q) if isinstance(q, (list, tuple)) else len(q.keys())
        print(f"[check_api] quantile output OK ({n} quantiles)")
    except Exception as e:
        print(f"[check_api] WARNING: quantile output failed ({e}) — H4 needs it; "
              "check the client version / API docs")

    print()
    print("[check_api] cost quote (rule of thumb: tokens ~ rows x columns x ~1):")
    for n_rows in (401_125, 200_000, 100_000, 50_000):
        cells = n_rows * 52
        print(f"  context {n_rows:>7,} rows x 52 cols ~ {cells/1e6:6.1f}M cells "
              f"per fit+predict call")
    print("  Account default: 5M tokens/day, 20M/month; hackathon entrants can request "
          "extra credits on the Prior Labs platform.")
    print("  Pick the PRIMARY context as the largest of {all, 200k, 100k, 50k} that "
          "fits the quota — before looking at any result (see PREREGISTRATION.md).")
    print("[check_api] OK")


if __name__ == "__main__":
    main()
