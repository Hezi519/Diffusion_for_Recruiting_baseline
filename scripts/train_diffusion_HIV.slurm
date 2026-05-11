#!/bin/bash
#SBATCH --job-name=diff_HIV
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=seas_gpu

#SBATCH --chdir=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline
#SBATCH --output=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline/logs/diff_HIV_%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline/logs/diff_HIV_%j.err

#SBATCH --mail-type=END
#SBATCH --mail-user=andrewma@g.harvard.edu

set -euo pipefail

WORK_DIR="/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline"
CHECKPOINT_DIR="$WORK_DIR/checkpoints/diffusion"
ENV_NAME="dpmd_rf"

mkdir -p "$WORK_DIR/logs" "$CHECKPOINT_DIR"
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
echo "HOST: $(hostname)"
echo "PWD:  $(pwd)"
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

python3 -u -m src.scripts.train_diffusion \
  --data_dir ICPSR_22140 \
  --std_name HIV \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --seed 42

echo "[DONE]"
