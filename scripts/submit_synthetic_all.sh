#!/bin/bash
# Submit all 4 synthetic-env methods as independent SLURM jobs.
#
# Each method runs as its own sbatch so they can fail/finish independently
# and you can resubmit just the one you want without recomputing the others.
# Per-method outputs land in $WORK_DIR/results/synthetic/<env_type>/<method>/
# so there is no race between concurrent jobs.
#
# Usage:
#   ./scripts/submit_synthetic_all.sh                       # basic env, seed 42
#   ENV_TYPE=tunnel_vision ./scripts/submit_synthetic_all.sh
#   SEED=7 TRAIN_EP=1000 ./scripts/submit_synthetic_all.sh
#   METHODS="dqn structured" ./scripts/submit_synthetic_all.sh   # subset

set -euo pipefail

ENV_TYPE="${ENV_TYPE:-basic}"
SEED="${SEED:-42}"
BUDGET="${BUDGET:-100}"
TRAIN_EP="${TRAIN_EP:-500}"
EVAL_EP="${EVAL_EP:-20}"
DISCOUNT="${DISCOUNT:-0.9}"
GAMMA="${GAMMA:-$DISCOUNT}"
METHODS="${METHODS:-random dqn structured gfp adaptive}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting synthetic-env jobs"
echo "  ENV_TYPE = $ENV_TYPE"
echo "  SEED     = $SEED"
echo "  BUDGET   = $BUDGET"
echo "  TRAIN_EP = $TRAIN_EP"
echo "  EVAL_EP  = $EVAL_EP"
echo "  DISCOUNT = $DISCOUNT"
echo "  GAMMA    = $GAMMA"
echo "  METHODS  = $METHODS"
echo ""

for METHOD in $METHODS; do
  SCRIPT="$SCRIPT_DIR/synthetic_${METHOD}.slurm"
  if [[ ! -f "$SCRIPT" ]]; then
    echo "  [skip] $METHOD: $SCRIPT not found" >&2
    continue
  fi
  JOB=$(sbatch \
    --export=ALL,ENV_TYPE="$ENV_TYPE",SEED="$SEED",BUDGET="$BUDGET",TRAIN_EP="$TRAIN_EP",EVAL_EP="$EVAL_EP",DISCOUNT="$DISCOUNT",GAMMA="$GAMMA" \
    "$SCRIPT" | awk '{print $NF}')
  echo "  [$METHOD] submitted job $JOB  ($SCRIPT)"
done

echo ""
echo "Check status:    squeue -u \$USER"
echo "Logs land in:    \$WORK_DIR/logs/syn_<method>_B${BUDGET}_F10_<jobid>.{out,err}"
echo "Results land in: \$WORK_DIR/results/synthetic/${ENV_TYPE}_b${BUDGET}/<method>/"
