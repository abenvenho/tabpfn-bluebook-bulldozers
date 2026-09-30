"""Scoreboard and pre-registered verdicts (H1–H5).

Usage: python -m src.compare [--primary <tag>]
Reads results/metrics/*.json and results/preds/*.csv; writes results/scoreboard.md and
results/verdicts.md. Hypotheses whose inputs are missing are marked "not yet testable".
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from .metrics import bootstrap_rmsle_diff

H1_H2_MARGIN = 0.005          # RMSLE, pre-registered
H4_RANGE = (0.75, 0.85)       # 80% interval coverage band
H5_THRESHOLD = 0.25339        # top 10% of the 2013 final leaderboard (47th of 477)


def load_metrics():
    rows = []
    for p in sorted(glob.glob("results/metrics/*.json")):
        if p.endswith("noise_audit.json"):
            continue
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def preds(tag):
    p = Path(f"results/preds/{tag}.csv")
    return pd.read_csv(p) if p.exists() else None


def paired(tag_a, tag_b):
    """Bootstrap RMSLE(A) − RMSLE(B) on the sales both runs scored."""
    a, b = preds(tag_a), preds(tag_b)
    if a is None or b is None or "y_true" not in a.columns:
        return None
    m = a.merge(b[["SalesID", "pred"]], on="SalesID", suffixes=("_a", "_b"))
    if len(m) == 0:
        return None
    return bootstrap_rmsle_diff(m["y_true"], m["pred_a"], m["pred_b"])


def fmt_ci(r):
    return (f"{r['diff']:+.5f} (95% CI {r['ci_low']:+.5f} .. {r['ci_high']:+.5f})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="primary_raw",
                    help="tag of the primary TabPFN run (raw arm)")
    args = ap.parse_args(argv)

    rows = load_metrics()
    if not rows:
        print("[compare] no metrics found under results/metrics/")
        return

    # ---------- scoreboard ----------
    df = pd.DataFrame(rows)
    for c in ["arm", "context", "sampling", "mode", "n_context", "coverage80"]:
        if c not in df.columns:
            df[c] = None
    df = df.sort_values("rmsle", na_position="last")
    cols = ["tag", "split", "arm", "context", "sampling", "mode", "n_context",
            "rmsle", "coverage80", "wall_seconds"]
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].copy()
    fmt = {"n_context": "{:.0f}", "rmsle": "{:.5f}", "coverage80": "{:.1%}",
           "wall_seconds": "{:.1f}"}
    for c, f in fmt.items():
        if c in disp.columns:
            disp[c] = disp[c].map(lambda v: "—" if pd.isna(v) else f.format(v))
    disp = disp.fillna("—")
    lines = ["# Scoreboard", "", disp.to_markdown(index=False), ""]
    Path("results/scoreboard.md").write_text("\n".join(lines))
    print("[compare] scoreboard -> results/scoreboard.md")

    # ---------- helpers ----------
    def meta(tag):
        return next((r for r in rows if r.get("tag") == tag), None)

    def find(**kw):
        out = []
        for r in rows:
            if all(str(r.get(k)) == str(v) for k, v in kw.items()):
                out.append(r)
        return out

    v = ["# Pre-registered verdicts", "",
         f"Primary run: `{args.primary}`. Margin (H1, H2): {H1_H2_MARGIN} RMSLE. "
         f"H5 threshold: {H5_THRESHOLD}.", ""]
    prim = meta(args.primary)

    # ---------- H1: raw TabPFN vs engineered LightGBM ----------
    r = paired(args.primary, "lgbm_engineered") if prim else None
    if r is None:
        v.append("## H1 — not yet testable (needs the primary run and lgbm_engineered)")
    else:
        refuted = r["diff"] > H1_H2_MARGIN and r["ci_low"] > 0
        v.append(f"## H1 — {'REFUTED' if refuted else 'corroborated'}")
        v.append(f"RMSLE(TabPFN raw) − RMSLE(LightGBM) = {fmt_ci(r)}; "
                 f"refuted iff diff > {H1_H2_MARGIN} with CI above 0.")
    v.append("")

    # ---------- H2: cleaning / appendix do not improve TabPFN ----------
    if prim:
        for arm in ["clean", "appendix"]:
            cands = find(split="valid", arm=arm, mode=prim.get("mode"),
                         context=prim.get("context"))
            if not cands:
                v.append(f"## H2 ({arm}) — not yet testable")
                continue
            r = paired(args.primary, cands[0]["tag"])
            if r is None:
                v.append(f"## H2 ({arm}) — not yet testable")
                continue
            # improvement of the treated arm = raw worse than treated = diff > 0
            refuted = r["diff"] > H1_H2_MARGIN and r["ci_low"] > 0
            v.append(f"## H2 ({arm}) — {'REFUTED' if refuted else 'corroborated'}")
            v.append(f"RMSLE(raw) − RMSLE({arm}) = {fmt_ci(r)}; refuted iff the {arm} arm "
                     f"improves by more than {H1_H2_MARGIN} (CI above 0).")
    else:
        v.append("## H2 — not yet testable (primary run missing)")
    v.append("")

    # ---------- H3: recent context beats random context at equal size ----------
    pairs, seen = [], set()
    for r_ in find(split="valid", arm="raw", sampling="recent"):
        if r_.get("context") in ("all", None) or r_["context"] in seen:
            continue
        rand = find(split="valid", arm="raw", sampling="random",
                    context=r_["context"], mode=r_.get("mode"))
        if rand:
            seen.add(r_["context"])
            pairs.append((r_, rand[0]))
    if not pairs:
        v.append("## H3 — not yet testable (needs recent vs random at equal context size)")
    else:
        for rec, ran in pairs:
            r = paired(rec["tag"], ran["tag"])
            if r is None:
                continue
            refuted = r["diff"] >= 0  # recent not better
            v.append(f"## H3 (context {rec['context']}) — "
                     f"{'REFUTED' if refuted else 'corroborated'}")
            v.append(f"RMSLE(recent) − RMSLE(random) = {fmt_ci(r)}; refuted iff ≥ 0.")
    v.append("")

    # ---------- H4: 80% interval coverage within 75–85% ----------
    if prim and prim.get("coverage80") is not None:
        c = prim["coverage80"]
        ok = H4_RANGE[0] <= c <= H4_RANGE[1]
        v.append(f"## H4 — {'corroborated' if ok else 'REFUTED'}")
        v.append(f"80% interval coverage on the validation split: {c:.1%} "
                 f"(band {H4_RANGE[0]:.0%}–{H4_RANGE[1]:.0%}).")
    else:
        v.append("## H4 — not yet testable (primary run has no quantiles yet)")
    v.append("")

    # ---------- H5: 2013 top-10% reference (late submissions are closed) ----------
    ks = Path("results/kaggle_score.txt")
    if ks.exists():
        score = float(ks.read_text().strip())
        refuted = score > H5_THRESHOLD
        v.append(f"## H5 — {'REFUTED' if refuted else 'corroborated'}")
        v.append(f"Kaggle private score {score:.5f} vs threshold {H5_THRESHOLD} "
                 f"(47th of 477 in 2013; supersedes the validation-split comparison).")
    elif prim and prim.get("rmsle") is not None:
        score = float(prim["rmsle"])
        refuted = score > H5_THRESHOLD
        v.append(f"## H5 — {'REFUTED' if refuted else 'corroborated'}")
        v.append(f"Primary validation RMSLE {score:.5f} vs {H5_THRESHOLD} (top 10% of the "
                 f"2013 final leaderboard). Pre-registered caveat: leaderboard-mirror "
                 f"split, not a like-for-like rank — Kaggle late submissions are closed.")
    else:
        v.append("## H5 — not yet testable (needs the primary run)")
    v.append("")

    Path("results/verdicts.md").write_text("\n".join(v))
    print("[compare] verdicts -> results/verdicts.md")


if __name__ == "__main__":
    main()
