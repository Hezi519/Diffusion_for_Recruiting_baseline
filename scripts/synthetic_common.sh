#!/bin/bash
# Shared environment + setup for synthetic_driver SLURM jobs.
# Sourced by each per-method slurm script. Keeps cluster paths / conda / debug
# print-out in one place.
#
# Inputs (env vars expected to be set by the caller):
#   WORK_DIR       — repo root on netscratch
#   ENV_NAME       — conda env name
#   RESULTS_ROOT   — top-level results dir (will be created)
#   METHOD         — one of: random, dqn, structured, gfp  (for logs/results subdir)
#   ENV_TYPE       — basic | tunnel_vision  (default: basic)

set -euo pipefail

: "${WORK_DIR:?WORK_DIR not set}"
: "${ENV_NAME:?ENV_NAME not set}"
: "${RESULTS_ROOT:?RESULTS_ROOT not set}"
: "${METHOD:?METHOD not set}"
ENV_TYPE="${ENV_TYPE:-basic}"

mkdir -p "$WORK_DIR/logs" "$RESULTS_ROOT/$ENV_TYPE/$METHOD"
cd "$WORK_DIR"

module purge
module load cuda/12.4.1-fasrc01 2>/dev/null || true

if [ -f "/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh" ]; then
  source "/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh"
elif [ -f "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh" ]; then
  source "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh"
else
  echo "ERROR: Could not find site Miniforge conda.sh" >&2
  exit 1
fi

conda activate "$ENV_NAME"

echo "================ JOB DEBUG ================"
echo "HOST:        $(hostname)"
echo "PWD:         $(pwd)"
echo "METHOD:      $METHOD"
echo "ENV_TYPE:    $ENV_TYPE"
echo "RESULTS_DIR: $RESULTS_ROOT/$ENV_TYPE/$METHOD"
which python3
python3 --version
python3 - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi || true
echo "=========================================="
