#!/bin/bash
#SBATCH --job-name=syn_dqn_B500_F10
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=seas_gpu

#SBATCH --chdir=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline
#SBATCH --output=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline/logs/syn_dqn_B500_F10_%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline/logs/syn_dqn_B500_F10_%j.err

#SBATCH --mail-type=END
#SBATCH --mail-user=andrewma@g.harvard.edu

# Method:  dqn  (Budget-DQN + GreedyAlloc — myopic RL baseline; "rl model")
# Env:     synthetic basic env (random covariate pool)

export WORK_DIR="/n/netscratch/tambe_lab/Lab/anma/Diffusion_for_Recruiting_baseline"
export ENV_NAME="dpmd_rf"
export RESULTS_ROOT="$WORK_DIR/results/synthetic"
export METHOD="dqn"
export ENV_TYPE="${ENV_TYPE:-basic}"

source "$WORK_DIR/scripts/synthetic_common.sh"

SEED="${SEED:-42}"
TRAIN_EP="${TRAIN_EP:-500}"
EVAL_EP="${EVAL_EP:-20}"

python3 -u -m src.scripts.synthetic_driver \
  --env_type "$ENV_TYPE" \
  --budget 500 \
  --initial_frontier_size 10 \
  --discount 0.9 \
  --max_rounds 50 \
  --train_episodes "$TRAIN_EP" \
  --n_episodes_eval "$EVAL_EP" \
  --model_mode learned \
  --methods dqn \
  --seed "$SEED" \
  --device cuda \
  --results_dir "$RESULTS_ROOT/$ENV_TYPE/$METHOD"

echo "[DONE]"
