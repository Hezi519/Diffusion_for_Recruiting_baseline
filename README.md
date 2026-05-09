# Diffusion Adaptive Surrogate

This folder connects the original adaptive surrogate allocation algorithm to
`Diffusion_for_Recruiting_baseline-leakage-free-split` without modifying that
project.

## What It Does

- Fits the existing leakage-free `GaussianCountModel`.
- Converts each GPR prediction into a discrete recruit-capacity distribution.
- Precomputes the adaptive surrogate value table `U(r, n)`.
- Evaluates the policy inside the existing diffusion `RecruitingEnv`.

## Run

```bash
python /home/u/diffusion_surrogatet/evaluate_adaptive_surrogate.py \
  --project_dir /home/u/Diffusion_for_Recruiting_baseline-leakage-free-split \
  --model_path /home/u/Diffusion_for_Recruiting_baseline-leakage-free-split/checkpoints/diffusion/ddpm_HIV.pt \
  --data_dir /home/u/Diffusion_for_Recruiting_baseline-leakage-free-split/ICPSR_22140 \
  --std_name HIV \
  --budget 100 \
  --initial_frontier_size 10 \
  --n_episodes_eval 10 \
  --discount 0.9
```

The surrogate table is cached in `/home/u/diffusion_surrogatet/cache`, and
evaluation vectors, trajectories, and plots are written to
`/home/u/diffusion_surrogatet/results`.

For larger budgets such as 500, the dynamic program can take noticeably longer.
The cache avoids recomputing it on repeated runs with the same settings.
