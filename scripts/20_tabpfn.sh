#!/usr/bin/env bash
# TabPFN runs, one block per plan day. Usage: scripts/20_tabpfn.sh <block> [context]
#   pilot    — raw arm, most recent 50k, validation split
#   primary  — raw arm, PRIMARY context (arg 2 or $PRIMARY_CTX, default all)
#   arms     — clean + appendix arms at the primary context (H2)
#   context  — recent vs random at 200k/100k/50k (H3)
#   modes    — fast at the primary context; thinking at 200k (API cap for thinking)
#   kaggle   — raw arm on Test.csv -> late-submission file (H5)
# MOCK=1 uses the offline mock predictor (smoke test only).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
MOCKFLAG=${MOCK:+--mock}
CTX=${2:-${PRIMARY_CTX:-all}}
RUN="$PY -m src.tabpfn_runner"

case "${1:-}" in
  pilot)
    $RUN --split valid --arm raw --context 50000 --sampling recent --tag pilot_raw_50k $MOCKFLAG ;;
  primary)
    $RUN --split valid --arm raw --context "$CTX" --sampling recent --tag primary_raw $MOCKFLAG ;;
  arms)
    $RUN --split valid --arm clean    --context "$CTX" --sampling recent $MOCKFLAG
    $RUN --split valid --arm appendix --context "$CTX" --sampling recent $MOCKFLAG ;;
  context)
    for N in 200000 100000 50000; do
      $RUN --split valid --arm raw --context "$N" --sampling recent $MOCKFLAG
      $RUN --split valid --arm raw --context "$N" --sampling random $MOCKFLAG
    done ;;
  modes)
    $RUN --split valid --arm raw --context "$CTX" --sampling recent --mode fast     $MOCKFLAG
    $RUN --split valid --arm raw --context 200000 --sampling recent --mode thinking $MOCKFLAG ;;
  kaggle)
    $RUN --split test --arm raw --context "$CTX" --sampling recent --tag kaggle_primary $MOCKFLAG ;;
  *)
    echo "usage: scripts/20_tabpfn.sh {pilot|primary|arms|context|modes|kaggle} [context]"; exit 1 ;;
esac
