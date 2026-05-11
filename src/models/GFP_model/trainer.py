from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.environment.state import RecruitingState
from src.models.GFP_model.planner import GFPPlanner


@dataclass
class GFPTrainerConfig:
    train_iterations: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    target_update_interval: int = 25
    max_steps_per_episode: int = 50
    state_pool_size: int = 256
    random_rollout_episodes: int = 64


class GFPTrainer:
    """
    Fitted value iteration for the GFP frontier-value surrogate.

    Training targets use the same deterministic Laplace backup as planning:
    max_s Q_hat(r, F, s). The target planner is periodically synced from the
    online value surrogate for stability.
    """

    def __init__(
        self,
        env,
        initial_frontier_fn,
        value_surrogate,
        survival_model,
        laplace_provider,
        cfg: GFPTrainerConfig,
        gamma: float = 0.9,
        max_budget_per_round: int | None = None,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        self.env = env
        self.initial_frontier_fn = initial_frontier_fn
        self.value_surrogate = value_surrogate
        self.survival_model = survival_model
        self.laplace_provider = laplace_provider
        self.cfg = cfg
        self.gamma = float(gamma)
        self.max_budget_per_round = max_budget_per_round
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        import copy

        self.target_value_surrogate = copy.deepcopy(value_surrogate).to(self.device)
        self.target_laplace_provider = laplace_provider.fork_for_value_surrogate(
            self.target_value_surrogate,
            seed=seed + 991,
        )
        self.target_planner = GFPPlanner(
            value_surrogate=self.target_value_surrogate,
            survival_model=self.survival_model,
            laplace_provider=self.target_laplace_provider,
            gamma=self.gamma,
            max_budget_per_round=self.max_budget_per_round,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(self.value_surrogate.parameters(), lr=cfg.lr)
        self._sync_target()

    def _sync_target(self) -> None:
        self.target_value_surrogate.load_state_dict(self.value_surrogate.state_dict())
        self.target_value_surrogate.eval()
        self.target_laplace_provider.clear_cache()
        if hasattr(self.target_laplace_provider, "refresh"):
            self.target_laplace_provider.refresh()

    def _random_action(self, state: RecruitingState) -> np.ndarray:
        n = state.frontier_size
        if n == 0 or state.budget_remaining <= 0:
            return np.zeros(n, dtype=int)

        max_spend = int(state.budget_remaining)
        if self.max_budget_per_round is not None:
            max_spend = min(max_spend, int(self.max_budget_per_round))
        spend = int(self.rng.integers(0, max_spend + 1))
        action = np.zeros(n, dtype=int)
        for _ in range(spend):
            action[int(self.rng.integers(0, n))] += 1
        return action

    def _collect_state_pool(self) -> list[RecruitingState]:
        states: list[RecruitingState] = []
        for _ in range(max(1, self.cfg.random_rollout_episodes)):
            state = self.env.reset(self.initial_frontier_fn())
            states.append(state)
            for _ in range(self.cfg.max_steps_per_episode):
                action = self._random_action(state)
                state, _, done, _ = self.env.step(action)
                states.append(state)
                if done or len(states) >= self.cfg.state_pool_size:
                    break
            if len(states) >= self.cfg.state_pool_size:
                break
        return states

    @torch.no_grad()
    def _backup_target(self, state: RecruitingState) -> float:
        if state.budget_remaining <= 0 or state.frontier_size == 0:
            return 0.0
        return self.target_planner.plan(state).value

    def train(self) -> dict:
        self.value_surrogate.train()
        pool = self._collect_state_pool()
        if not pool:
            return {"final_loss": 0.0, "state_pool_size": 0}

        losses: list[float] = []
        for it in range(self.cfg.train_iterations):
            idx = self.rng.integers(0, len(pool), size=max(1, self.cfg.batch_size))
            batch_states = [pool[int(i)] for i in idx]

            preds = []
            targets = []
            for state in batch_states:
                covariates = torch.tensor(
                    state.frontier_covariates,
                    dtype=torch.float32,
                    device=self.device,
                )
                pred = self.value_surrogate(covariates, float(state.budget_remaining))
                target = torch.tensor(
                    self._backup_target(state),
                    dtype=torch.float32,
                    device=self.device,
                )
                preds.append(pred)
                targets.append(target)

            loss = F.mse_loss(torch.stack(preds), torch.stack(targets))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_surrogate.parameters(), 10.0)
            self.optimizer.step()

            losses.append(float(loss.item()))
            if (it + 1) % max(1, self.cfg.target_update_interval) == 0:
                self._sync_target()
                self.laplace_provider.clear_cache()
                if hasattr(self.laplace_provider, "refresh"):
                    self.laplace_provider.refresh()

        self.value_surrogate.eval()
        self.laplace_provider.clear_cache()
        if hasattr(self.laplace_provider, "refresh"):
            self.laplace_provider.refresh()
        return {
            "final_loss": losses[-1] if losses else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "state_pool_size": len(pool),
        }
