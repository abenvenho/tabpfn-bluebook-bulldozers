#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (53-column Kaggle schema). No API, no token:
# the TabPFN calls use the offline mock. Exercises prepare, baselines, all arms,
# H3 context pair, the Kaggle submission path, compare and figures.
# It runs in a temporary copy of the repository, so the versioned results/ are never
# overwritten. SMOKE_KEEP=1 keeps that copy for inspection.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${SMOKE_IN_COPY:-}" ]; then
  ROOT="$PWD"
  PYTHON=${PYTHON:-$([ -x .venv/bin/python ] && echo "$ROOT/.venv/bin/python" || command -v python3)}
  case "$PYTHON" in /*) ;; *) [ -x "$PYTHON" ] && PYTHON="$ROOT/$PYTHON" ;; esac
  WORK=$(mktemp -d "${TMPDIR:-/tmp}/bluebook-smoke.XXXXXX")
  if [ -z "${SMOKE_KEEP:-}" ]; then trap 'rm -rf "$WORK"' EXIT; fi
  "$PYTHON" - "$ROOT" "$WORK" <<'EOF'
import os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
def ignore(d, names):
    if os.path.samefile(d, src):  # top level: no git, environments, data or results
        return [n for n in names if n in {".git", "data", "results"} or n.startswith(".venv")]
    return [n for n in names if n == "__pycache__"]
shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
EOF
  mkdir -p "$WORK/data/raw" "$WORK/results"
  echo "== smoke test in a temporary copy: $WORK (results/ here is untouched)"
  (cd "$WORK" && PYTHON="$PYTHON" SMOKE_IN_COPY=1 bash scripts/90_smoke.sh)
  exit
fi
PY=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
export MOCK=1

echo "== [1/6] synthetic data =="
$PY scripts/05_make_synthetic.py --n-train 6000 --n-valid 600 --n-test 600 --out data/synthetic

echo "== [2/6] prepare + noise audit =="
$PY scripts/01_prepare.py --raw-dir data/synthetic --out-dir data/prepared

echo "== [3/6] baselines =="
bash scripts/10_baselines.sh

echo "== [4/6] tabpfn arms (mock) =="
bash scripts/20_tabpfn.sh primary
bash scripts/20_tabpfn.sh arms
$PY -m src.tabpfn_runner --split valid --arm raw --context 2000 --sampling recent --mock
$PY -m src.tabpfn_runner --split valid --arm raw --context 2000 --sampling random --mock

echo "== [5/6] kaggle submission path (mock) =="
bash scripts/20_tabpfn.sh kaggle

echo "== [6/6] compare + figures =="
$PY -m src.compare --primary primary_raw
$PY -m src.make_figures --primary primary_raw

for f in results/scoreboard.md results/verdicts.md \
         results/figures/rmsle_by_run.png results/figures/pred_vs_actual.png \
         results/figures/calibration.png results/submission_kaggle_primary.csv \
         results/metrics/noise_audit.json; do
  [ -s "$f" ] || { echo "SMOKE FAIL: missing $f"; exit 1; }
done
echo "SMOKE OK"
