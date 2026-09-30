#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (53-column Kaggle schema). No API, no token:
# the TabPFN calls use the offline mock. Exercises prepare, baselines, all arms,
# H3 context pair, the Kaggle submission path, compare and figures.
set -euo pipefail
cd "$(dirname "$0")/.."
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
