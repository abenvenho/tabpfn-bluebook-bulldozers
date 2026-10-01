"""Figures for the README (matplotlib PNGs under results/figures/).

Usage: python -m src.make_figures [--primary <tag>]
Palette: validated colorblind-safe defaults (blue #2a78d6 for TabPFN runs, neutral
gray for baselines, ink #0b0b0b / #52514e); one axis per chart; direct labels.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, GRAY = "#2a78d6", "#b0afa8"
ORANGE = "#eb6834"  # categorical slot 2; blue/orange pair validated (CVD dE 24.7)
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e5e4df"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)


def load_metrics():
    rows = []
    for p in sorted(glob.glob("results/metrics/*.json")):
        if p.endswith("noise_audit.json"):
            continue
        with open(p) as f:
            r = json.load(f)
        if r.get("split") == "valid" and r.get("rmsle") is not None:
            rows.append(r)
    return rows


ARM_LABEL = {"raw": "raw table", "clean": "2013 cleaning",
             "appendix": "raw + Machine Appendix"}


def _label(r):
    """Readable name of a run for the figures."""
    tag = str(r.get("tag"))
    if tag == "lgbm_engineered":
        return "LightGBM, 2013-style engineering"
    if tag == "median_benchmark":
        return "median benchmark"
    arm = ARM_LABEL.get(r.get("arm"), str(r.get("arm")))
    if r.get("context") in ("all", None):
        ctx = "all sales"
    else:
        ctx = (f"{int(r['n_context']) // 1000}k "
               f"{'most recent' if r.get('sampling') == 'recent' else 'random'}")
    mode = "" if r.get("mode") in ("base", None) else f", {r['mode']} mode"
    extra = " (primary)" if tag == "primary_raw" else ""
    return f"{arm}, {ctx}{mode}{extra}"


def _title_name(tag):
    return "TabPFN-3.5, raw table, all sales" if tag == "primary_raw" else tag


def _is_tabpfn(r):
    p = str(r.get("predictor", ""))
    return "tabpfn" in p or "mock" in p


def fig_rmsle_by_run(rows, out):
    """Dot plot, one row per distinct configuration. The pilot repeats the 50k
    most-recent configuration with identical predictions, so it is not drawn
    twice; the median benchmark is off scale and named in the axis label."""
    keep = [r for r in rows if r.get("tag") not in ("median_benchmark", "pilot_raw_50k")]
    med = next((r["rmsle"] for r in rows if r.get("tag") == "median_benchmark"), None)
    lgbm = next((r["rmsle"] for r in rows if r.get("tag") == "lgbm_engineered"), None)
    df = pd.DataFrame(keep).sort_values("rmsle", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(df) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(False, axis="y")
    y = np.arange(len(df))
    if lgbm is not None:
        ax.axvline(lgbm, color=INK2, linewidth=1.2, linestyle="--", zorder=2)
    span = df["rmsle"].max() - df["rmsle"].min()
    for yi, (_, r) in zip(y, df.iterrows()):
        color = BLUE if _is_tabpfn(r) else GRAY
        ax.plot(r["rmsle"], yi, "o", markersize=8, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(r["rmsle"] + span * 0.03, yi, f"{r['rmsle']:.4f}", va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks(y, [_label(r) for _, r in df.iterrows()], fontsize=8.5)
    ax.set_xlim(df["rmsle"].min() - span * 0.08, df["rmsle"].max() + span * 0.22)
    xlabel = "RMSLE on the validation split (Jan–Apr 2012), lower is better"
    if med is not None:
        xlabel += f"\nmedian benchmark {med:.4f}, off scale"
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_title("Out-of-time error by configuration", color=INK, fontsize=11, loc="left")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=BLUE),
               plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=GRAY),
               plt.Line2D([], [], color=INK2, linewidth=1.2, linestyle="--")]
    ax.legend(handles, ["TabPFN-3.5", "LightGBM baseline", "LightGBM level"],
              frameon=True, facecolor=SURFACE, edgecolor=SURFACE, framealpha=1,
              fontsize=8.5, loc="upper right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_context(rows, primary_tag, out):
    """H3: RMSLE by context size, most recent vs random sample. Both samplings
    coincide at 'all sales', so both lines end on the primary run."""
    base = [r for r in rows if r.get("arm") == "raw" and r.get("mode") == "base"
            and r.get("predictor") == "tabpfn_client"]
    prim = next((r for r in rows if r.get("tag") == primary_tag), None)
    series = {}
    for sampling in ("recent", "random"):
        pts = {}
        for r in base:
            if r.get("sampling") == sampling and r.get("context") not in ("all", None):
                pts[int(r["n_context"])] = r["rmsle"]  # duplicates are identical runs
        if prim is not None:
            pts[int(prim["n_context"])] = prim["rmsle"]
        series[sampling] = sorted(pts.items())
    if not all(len(v) >= 2 for v in series.values()):
        return False
    lgbm = next((r["rmsle"] for r in rows if r.get("tag") == "lgbm_engineered"), None)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    if lgbm is not None:
        ax.axhline(lgbm, color=INK2, linewidth=1.2, linestyle="--", zorder=2)
        ax.text(390_000, lgbm + 0.0003, f"engineered LightGBM  {lgbm:.4f}",
                color=INK2, fontsize=8.5, va="bottom", ha="right")
    for sampling, color, label in (("recent", BLUE, "most recent sales"),
                                   ("random", ORANGE, "random sample")):
        xs, ys = zip(*series[sampling])
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label=label)
        # direct label at the 200k point, where the two lines are furthest apart
        if 200_000 in xs:
            y = ys[xs.index(200_000)]
            if sampling == "recent":
                ax.annotate(f"{y:.4f}", (200_000, y), textcoords="offset points",
                            xytext=(8, 8), fontsize=8.5, color=INK)
            else:
                ax.annotate(f"{y:.4f}", (200_000, y), textcoords="offset points",
                            xytext=(-8, -14), fontsize=8.5, color=INK, ha="right")
    if prim is not None:
        ax.annotate(f"all {int(prim['n_context']):,} sales  {prim['rmsle']:.4f}",
                    (prim["n_context"], prim["rmsle"]), textcoords="offset points",
                    xytext=(-10, -14), fontsize=8.5, color=INK, ha="right")
    lo = min(y for v in series.values() for _, y in v)
    hi = max([y for v in series.values() for _, y in v] + ([lgbm] if lgbm else []))
    pad = (hi - lo) * 0.12
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xscale("log")
    ticks = sorted({x for v in series.values() for x, _ in v})
    ax.set_xticks(ticks, [f"{t // 1000}k" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Context size (training sales handed to TabPFN), log scale",
                  color=INK2, fontsize=9)
    ax.set_ylabel("RMSLE, Jan–Apr 2012 (lower is better)", color=INK2, fontsize=9)
    ax.set_title("H3 — recency helps small contexts, coverage wins at 200k",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_pred_vs_actual(tag, out):
    p = Path(f"results/preds/{tag}.csv")
    if not p.exists():
        return False
    df = pd.read_csv(p)
    if "y_true" not in df.columns:
        return False
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    lo = min(df["y_true"].min(), df["pred"].min()) * 0.9
    hi = max(df["y_true"].max(), df["pred"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1.2, linestyle="--", zorder=2)
    ax.scatter(df["y_true"], df["pred"], s=6, alpha=0.25, color=BLUE, lw=0, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Auction price (USD, log scale)", color=INK2, fontsize=9)
    ax.set_ylabel("Predicted price (USD, log scale)", color=INK2, fontsize=9)
    ax.set_title(f"Predicted vs realized, Jan–Apr 2012\n{_title_name(tag)}", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_calibration(tag, out):
    p = Path(f"results/preds/{tag}.csv")
    if not p.exists():
        return False
    df = pd.read_csv(p)
    qcols = [c for c in df.columns if c.startswith("q")]
    if "y_true" not in df.columns or not qcols:
        return False
    nominal = [int(c[1:]) / 100 for c in qcols]
    empirical = [float((df["y_true"] <= df[c]).mean()) for c in qcols]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1.2, linestyle="--", zorder=2)
    ax.plot(nominal, empirical, color=BLUE, linewidth=2, marker="o", markersize=6,
            zorder=3)
    for x, ye in zip(nominal, empirical):
        ax.annotate(f"{ye:.0%}", (x, ye), textcoords="offset points", xytext=(6, -10),
                    fontsize=8.5, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal quantile", color=INK2, fontsize=9)
    ax.set_ylabel("Share of realized prices at or below it", color=INK2, fontsize=9)
    ax.set_title(f"Quantile calibration, four months ahead\n{_title_name(tag)}",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="primary_raw")
    args = ap.parse_args(argv)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    rows = load_metrics()
    if rows:
        fig_rmsle_by_run(rows, "results/figures/rmsle_by_run.png")
        print("[figures] results/figures/rmsle_by_run.png")
    if rows and fig_context(rows, args.primary, "results/figures/context_h3.png"):
        print("[figures] results/figures/context_h3.png")
    tag = args.primary
    if not Path(f"results/preds/{tag}.csv").exists() and rows:
        tag = max(rows, key=lambda r: r.get("rmsle") is not None and -r["rmsle"])["tag"]
    if fig_pred_vs_actual(tag, "results/figures/pred_vs_actual.png"):
        print("[figures] results/figures/pred_vs_actual.png")
    if fig_calibration(tag, "results/figures/calibration.png"):
        print("[figures] results/figures/calibration.png")


if __name__ == "__main__":
    main()
