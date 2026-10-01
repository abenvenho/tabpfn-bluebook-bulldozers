"""Exploratory analyses — NOT pre-registered.

Usage: python -m src.explore
Writes results/exploratory.md. Nothing here changes a verdict: the pre-registered
hypotheses are judged only by src/compare.py. These numbers describe the runs
that were outside the hypotheses (fast and thinking modes), the reproducibility
check that the context block made possible, the measured API spend, and the
server-side timings.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import bootstrap_rmsle_diff, rmsle

# Server quotes (tabpfn_client.estimate_cost, pricing quota_v3), transcribed in
# results/cost_quote_2026-10-01.txt. Per block = quote x number of predict calls.
QUOTES = {
    "arms": 732_938,
    "context": 326_828,
    "fast": 2 * 142_298,
    "kaggle": 365_272,
}


def _load(tag):
    p = Path(f"results/preds/{tag}.csv")
    return pd.read_csv(p) if p.exists() else None


def _paired(tag_a, tag_b):
    a, b = _load(tag_a), _load(tag_b)
    if a is None or b is None or "y_true" not in a.columns:
        return None
    m = a.merge(b[["SalesID", "pred"]], on="SalesID", suffixes=("_a", "_b"))
    r = bootstrap_rmsle_diff(m["y_true"], m["pred_a"], m["pred_b"])
    r["rmsle_a"] = rmsle(m["y_true"], m["pred_a"])
    r["rmsle_b"] = rmsle(m["y_true"], m["pred_b"])
    return r


def _metrics():
    rows = {}
    for p in sorted(glob.glob("results/metrics/*.json")):
        if p.endswith("noise_audit.json"):
            continue
        with open(p) as f:
            r = json.load(f)
        rows[r["tag"]] = r
    return rows


def main():
    rows = _metrics()
    out = ["# Exploratory analyses (not pre-registered)", "",
           "Nothing in this file changes a pre-registered verdict "
           "(see `verdicts.md`). Paired bootstrap as in the pre-registration "
           "(10,000 resamples, seed 42, percentile 95% CI); positive difference "
           "= the first run is worse.", ""]

    # ---------- modes ----------
    out += ["## TabPFN-3.5 modes", "",
            "| Comparison | RMSLE A | RMSLE B | A − B | 95% CI |",
            "|---|---|---|---|---|"]
    for label, a, b in [
        ("fast (all sales) vs base (all sales)",
         "tabpfn_fast_raw_valid_all_recent", "primary_raw"),
        ("thinking (50k recent) vs base (50k recent)",
         "tabpfn_thinking_raw_valid_50000_recent",
         "tabpfn_base_raw_valid_50000_recent"),
    ]:
        r = _paired(a, b)
        if r is None:
            out.append(f"| {label} | — | — | — | not available |")
            continue
        out.append(f"| {label} | {r['rmsle_a']:.5f} | {r['rmsle_b']:.5f} | "
                   f"{r['diff']:+.5f} | {r['ci_low']:+.5f} .. {r['ci_high']:+.5f} |")
    out += ["",
            "Thinking mode failed server-side twice at the pre-planned 200k context "
            "(\"The worker failed to process this request\"; request_ids "
            "`a724cd8f2ce84f9cabad7e6de3d06c74`, `ca766e3690b6410588dcc297d6439758`) "
            "and ran at 50k. It optimises RMSE of log1p(price) with its own internal "
            "validation; no `time_col` was passed, because the raw arm keeps the date "
            "as the CSV string and the API requires datetimes or numbers there.", ""]

    # ---------- beats LightGBM ----------
    lgbm = rows.get("lgbm_engineered", {}).get("rmsle")
    # the pilot repeats the 50k-recent configuration (identical predictions, see
    # below), so it is not counted twice
    tab = [r for r in rows.values() if r.get("predictor") == "tabpfn_client"
           and r.get("split") == "valid" and r.get("rmsle") is not None
           and r.get("tag") != "pilot_raw_50k"]
    if lgbm is not None and tab:
        better = [r for r in tab if r["rmsle"] < lgbm]
        worse = [r["tag"] for r in tab if r["rmsle"] >= lgbm]
        out += ["## Every TabPFN configuration against the engineered LightGBM", "",
                f"{len(better)} of {len(tab)} distinct TabPFN configurations on the "
                f"validation split have a lower RMSLE than the engineered LightGBM "
                f"({lgbm:.5f}); the pilot is not counted, as it repeats the 50k-recent "
                f"configuration. Not lower: {', '.join(f'`{t}`' for t in worse) or 'none'}. "
                "Point estimates; the paired test for the primary run is H1.", ""]

    # ---------- reproducibility ----------
    a, b = _load("pilot_raw_50k"), _load("tabpfn_base_raw_valid_50000_recent")
    if a is not None and b is not None:
        m = a.merge(b, on="SalesID", suffixes=("_1", "_2"))
        cols = [c for c in ("pred", "q10", "q25", "q50", "q75", "q90")
                if f"{c}_1" in m.columns]
        maxdiff = max(float((m[f"{c}_1"] - m[f"{c}_2"]).abs().max()) for c in cols)
        out += ["## Reproducibility of the API", "",
                f"The pilot (2026-09-30) and the 50k-recent run of the context block "
                f"(2026-10-01) use the same configuration. Over {len(m):,} validation "
                f"sales, the largest absolute difference across mean and "
                f"{', '.join(cols[1:])} is {maxdiff:.6g} USD.", ""]

    # ---------- spend ----------
    log = Path("results/usage_log.txt")
    if log.exists():
        rd = []
        for line in log.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2].startswith("used="):
                rd.append((parts[0], parts[1], int(parts[2].split("=")[1])))
        out += ["## Measured API spend (2026-10-01)", "",
                "Readings of the API's monthly usage counter (`results/usage_log.txt`) "
                "after each block, against the server quote.", "",
                "| Interval | Measured tokens | Quote | Note |", "|---|---|---|---|"]
        notes = {
            "apos-arms": ("arms", "counter assumed at 0 on Oct 1 (monthly reset); mid-block reading = first call's quote"),
            "apos-context": ("context", ""),
            "apos-fast": ("fast", "includes the first failed thinking fit (200k)"),
            "apos-kaggle": ("kaggle", "2x the quote; consistent with an automatic client retry, not confirmed"),
            "apos-thinking": (None, "second failed thinking fit (200k) + thinking 50k"),
        }
        prev = 0
        for ts, tag, used in rd:
            if tag == "durante-arms":
                continue
            block, note = notes.get(tag, (None, ""))
            q = QUOTES.get(block) if block else None
            out.append(f"| {tag} ({ts}) | {used - prev:,} | "
                       f"{'' if q is None else f'{q:,}'} | {note} |")
            prev = used
        out += ["", f"Total on 2026-10-01: {rd[-1][2]:,} tokens. Daily limit: 5,000,000 tokens, "
                "as stated by the API (`results/api_errors.txt`).", ""]

    # ---------- timings ----------
    trows = []
    for r in rows.values():
        t = r.get("server_timings")
        if not isinstance(t, dict):
            continue
        fit, pr = t.get("fit") or {}, t.get("predict") or {}
        trows.append((r["tag"], r.get("n_context"), r.get("n_eval"),
                      fit.get("train_set_transform_s"), fit.get("fit_s"),
                      pr.get("predict_s"), r.get("wall_seconds")))
    if trows:
        out += ["## Server-side timings (seconds)", "",
                "| Run | Context rows | Eval rows | Train-set transform | Fit | "
                "Predict (last call) | Wall (client) |",
                "|---|---|---|---|---|---|---|"]
        f = lambda v: "—" if v is None else f"{v:,.1f}" if isinstance(v, float) else f"{v:,}"
        for row in sorted(trows, key=lambda x: (-(x[1] or 0), x[0])):
            out.append("| " + " | ".join([f"`{row[0]}`"] + [f(v) for v in row[1:]]) + " |")
        out += ["", "Runs before the 2026-10-01 runner update (pilot, primary, arms) "
                "did not log server timings.", ""]

    Path("results/exploratory.md").write_text("\n".join(out))
    print("[explore] -> results/exploratory.md")


if __name__ == "__main__":
    main()
