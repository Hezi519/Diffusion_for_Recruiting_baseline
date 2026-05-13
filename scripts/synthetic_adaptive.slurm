#!/bin/bash
#SBATCH --job-name=syn_adaptive_B500_F10
#SBATCH --time=24:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=6
#SBATCH --partition=seas_compute

#SBATCH --chdir=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline
#SBATCH --output=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline/logs/syn_adaptive_B500_F10_%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline/logs/syn_adaptive_B500_F10_%j.err

#SBATCH --mail-type=END
#SBATCH --mail-user=andrewma@g.harvard.edu

# Method:  adaptive  (Poisson surrogate + DP precompute; no RL training)
# Env:     synthetic basic env (random covariate pool)
# No GPU needed — adaptive has no neural policy; only the learned count/DDPM
# offspring prelude runs on CPU when --model_mode learned.

export WORK_DIR="/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline"
export ENV_NAME="dpmd_rf"
export RESULTS_ROOT="$WORK_DIR/results/synthetic"
export ENV_TYPE="${ENV_TYPE:-basic}"

SEED="${SEED:-42}"
BUDGET="${BUDGET:-100}"
TRAIN_EP="${TRAIN_EP:-500}"
EVAL_EP="${EVAL_EP:-10}"
DISCOUNT="${DISCOUNT:-0.9}"

export METHOD="adaptive_disc${DISCOUNT}"

source "$WORK_DIR/scripts/synthetic_common.sh"

python3 -u -m src.scripts.synthetic_driver \
  --env_type "$ENV_TYPE" \
  --budget "$BUDGET" \
  --initial_frontier_size 10 \
  --discount "$DISCOUNT" \
  --max_rounds 50 \
  --train_episodes "$TRAIN_EP" \
  --n_episodes_eval "$EVAL_EP" \
  --model_mode learned \
  --methods adaptive \
  --seed "$SEED" \
  --device cpu \
  --results_dir "$RESULTS_ROOT/${ENV_TYPE}_b${BUDGET}/$METHOD"

echo "[DONE]"
