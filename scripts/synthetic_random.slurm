#!/bin/bash
#SBATCH --job-name=syn_random_B500_F10
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_compute

#SBATCH --chdir=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline
#SBATCH --output=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline/logs/syn_random_B500_F10_%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline/logs/syn_random_B500_F10_%j.err

#SBATCH --mail-type=END
#SBATCH --mail-user=andrewma@g.harvard.edu

# Method:  random  (uniform round-budget + multinomial split — sanity floor)
# Env:     synthetic basic env (random covariate pool)
# No GPU needed — random has no neural net.

export WORK_DIR="/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline"
export ENV_NAME="dpmd_rf"
export RESULTS_ROOT="$WORK_DIR/results/synthetic"
export ENV_TYPE="${ENV_TYPE:-basic}"

SEED="${SEED:-42}"
BUDGET="${BUDGET:-100}"
TRAIN_EP="${TRAIN_EP:-500}"
EVAL_EP="${EVAL_EP:-20}"
DISCOUNT="${DISCOUNT:-0.9}"

export METHOD="random_disc${DISCOUNT}"

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
  --methods random \
  --seed "$SEED" \
  --device cpu \
  --results_dir "$RESULTS_ROOT/${ENV_TYPE}_b${BUDGET}/$METHOD"

echo "[DONE]"
