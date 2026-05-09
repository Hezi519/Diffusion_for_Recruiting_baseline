#!/bin/bash
#SBATCH --job-name=srl_HIV_T100_B8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=seas_gpu

#SBATCH --chdir=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline
#SBATCH --output=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline/logs/srl_HIV_T100_B8_%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline/logs/srl_HIV_T100_B8_%j.err

#SBATCH --mail-type=END
#SBATCH --mail-user=andrewma@g.harvard.edu

set -euo pipefail

WORK_DIR="/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline"
RESULTS_DIR="$WORK_DIR/results_SRL_HIV"
MODEL_PATH="$WORK_DIR/checkpoints/diffusion/ddpm_HIV.pt"
ENV_NAME="dpmd_rf"

mkdir -p "$WORK_DIR/logs" "$RESULTS_DIR"
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

python3 -u -m src.scripts.structured_rl_driver \
  --model_path "$MODEL_PATH" \
  --device cuda \
  --std_name HIV \
  --train_episodes 200 \
  --n_episodes_eval 20 \
  --log_every 10 \
  --seed 42 \
  --results_dir "$RESULTS_DIR"

echo "[DONE]"
