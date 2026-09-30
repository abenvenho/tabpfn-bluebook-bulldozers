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


def fig_rmsle_by_run(rows, out):
    df = pd.DataFrame(rows).sort_values("rmsle", ascending=True)
    is_tabpfn = df["tag"].str.contains("tabpfn|primary", case=False) | df.get(
        "predictor", pd.Series([""] * len(df))).astype(str).str.contains("tabpfn|mock")
    colors = [BLUE if t else GRAY for t in is_tabpfn]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    y = np.arange(len(df))
    ax.barh(y, df["rmsle"], color=colors, height=0.62, zorder=3)
    for yi, v in zip(y, df["rmsle"]):
        ax.text(v + df["rmsle"].max() * 0.01, yi, f"{v:.4f}", va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks(y, df["tag"], fontsize=8.5)
    ax.set_xlabel("RMSLE on the validation split (Jan–Apr 2012) — lower is better",
                  color=INK2, fontsize=9)
    ax.set_title("Out-of-time error by run", color=INK, fontsize=11, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=GRAY)]
    ax.legend(handles, ["TabPFN", "baseline"], frameon=False, fontsize=8.5,
              loc="upper right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


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
    ax.set_title(f"Predicted vs realized — {tag}", color=INK, fontsize=11, loc="left")
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
    ax.set_title(f"Quantile calibration, four months ahead — {tag}",
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
    tag = args.primary
    if not Path(f"results/preds/{tag}.csv").exists() and rows:
        tag = max(rows, key=lambda r: r.get("rmsle") is not None and -r["rmsle"])["tag"]
    if fig_pred_vs_actual(tag, "results/figures/pred_vs_actual.png"):
        print("[figures] results/figures/pred_vs_actual.png")
    if fig_calibration(tag, "results/figures/calibration.png"):
        print("[figures] results/figures/calibration.png")


if __name__ == "__main__":
    main()
