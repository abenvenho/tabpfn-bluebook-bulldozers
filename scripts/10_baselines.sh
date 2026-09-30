#!/usr/bin/env bash
# Median benchmark + engineered LightGBM on the validation split.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
$PY -m src.baselines "$@"
