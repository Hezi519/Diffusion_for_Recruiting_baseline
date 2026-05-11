# Synthetic Environment — Branch Overview

This branch adds a self-contained synthetic recruiting environment that lets you run and evaluate the full pipeline (Random, Budget-DQN, Structured RL) without the real ICPSR dataset or any pretrained models.

## Why

The ICPSR dataset cannot be shared publicly, and fitting the DDPM covariate model is slow. The synthetic environment removes both dependencies, making it easy to iterate on algorithms and demonstrate key failure modes in isolation.

---

## Environments

### Basic (`--env_type basic`)

A general-purpose synthetic environment. Individuals are represented as random one-hot covariate vectors. Recruit rates are derived from covariates via a fixed random weight vector (no fitting required). Offspring covariates are sampled to resemble their recruiter, so investing in high-rate individuals compounds over rounds.

**Files:**
- `src/data/synthetic_generator.py` — random covariate pool
- `src/models/count_model/synthetic_count_model.py` — oracle Poisson count model
- `src/models/covariate_model/synthetic_covariate_model.py` — categorical inheritance model

---

### Tunnel Vision (`--env_type tunnel_vision`)

A designed environment to isolate and demonstrate the **tunnel-vision failure mode** of myopic allocation policies.

Individuals are one of three types, encoded in the LOCAL covariate group:

| Type | Recruit Rate | Offspring |
|------|-------------|-----------|
| A (boom-bust) | High | Type C (dead-end) |
| B (sustainable) | Lower | Type B (self-replicating) |
| C (dead-end) | Near zero | Type C |

A myopic agent (Budget-DQN) concentrates all budget on Type A because the immediate rate is highest. This yields a burst of recruits in round 1, but the entire frontier collapses to Type C dead-ends afterward. A farsighted agent (Structured RL) learns to invest in Type B despite the lower immediate rate, building a self-replicating chain that produces recruits across many rounds.

The `--max_budget_per_round` cap forces multi-round episodes, giving the Type B chain time to compound and making the gap between policies large and clear.

**Files:**
- `src/data/tunnel_vision_generator.py` — Type A/B/C covariate pool
- `src/models/count_model/tunnel_vision_count_model.py` — type-based oracle count model
- `src/models/covariate_model/tunnel_vision_covariate_model.py` — type-preserving offspring model
- `src/models/RL_model/tunnel_vision_greedy_allocator.py` — TypeA-only allocator (DQN's tunnel-vision policy)

---

## Driver

All experiments are run through a single entry point:

```bash
python -m src.scripts.synthetic_driver --env_type [basic|tunnel_vision] [options]
```

Key flags:

| Flag | Description |
|------|-------------|
| `--env_type` | `basic` or `tunnel_vision` |
| `--budget` | Total voucher budget per episode |
| `--initial_frontier_size` | Number of seed nodes |
| `--max_budget_per_round` | Cap on vouchers spent per round (tunnel vision only) |
| `--rate_a`, `--rate_b`, `--rate_c` | Type-specific Poisson rates (tunnel vision only) |
| `--methods` | Comma-separated list: `random`, `dqn`, `structured` |
| `--train_episodes` | Training episodes for RL methods |
| `--results_dir` | Where to save outputs |

### Example — Tunnel Vision

```bash
python -m src.scripts.synthetic_driver \
    --env_type tunnel_vision \
    --budget 100 \
    --initial_frontier_size 10 \
    --max_budget_per_round 10 \
    --rate_a 5.0 --rate_b 2.0 --rate_c 0.01 \
    --train_episodes 2000 \
    --methods random,dqn,structured \
    --results_dir results/tunnel_vision
```
