from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.environment.state import RecruitingState


@dataclass
class GFPPlanResult:
    budget: int
    allocation: np.ndarray
    value: float
    candidate_values: np.ndarray


class GFPPlanner:
    """
    Generative Frontier Planning from Algorithm 1 in the paper.

    For every feasible round budget s, greedily allocates s individual resource
    units by the marginal objective:

        immediate survival probability + discounted saturation continuation gain.
    """

    def __init__(
        self,
        value_surrogate,
        survival_model,
        laplace_provider,
        gamma: float = 0.9,
        max_budget_per_round: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.value_surrogate = value_surrogate
        self.survival_model = survival_model
        self.laplace_provider = laplace_provider
        self.gamma = float(gamma)
        self.max_budget_per_round = max_budget_per_round
        self.device = torch.device(device)

    @torch.no_grad()
    def _weights(self, budget_remaining: int) -> np.ndarray:
        w = self.value_surrogate.weights(float(budget_remaining))
        return w.detach().cpu().numpy().astype(np.float64)

    def _candidate_budget_limit(self, state: RecruitingState) -> int:
        limit = max(0, int(state.budget_remaining))
        if self.max_budget_per_round is not None:
            limit = min(limit, max(0, int(self.max_budget_per_round)))
        return limit

    def _greedy_for_budget(
        self,
        covariates: np.ndarray,
        round_budget: int,
        remaining_after_spend: int,
        alpha: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        n = covariates.shape[0]
        latent_dim = alpha.shape[1]
        allocation = np.zeros(n, dtype=int)
        immediate_value = 0.0
        u = np.ones(latent_dim, dtype=np.float64)
        weights = self._weights(remaining_after_spend)

        if n == 0 or round_budget <= 0:
            continuation = self.gamma * float(np.sum(weights * (1.0 - u)))
            return allocation, continuation

        for _ in range(round_budget):
            current_tau = self.survival_model.tau(covariates, alpha, allocation)
            next_alloc = allocation + 1
            next_tau = self.survival_model.tau(covariates, alpha, next_alloc)
            beta = np.clip(next_tau / np.clip(current_tau, 1e-12, 1.0), 1e-12, 1.0)

            next_resource = allocation + 1
            p = self.survival_model.survival_prob(covariates, next_resource)
            continuation_gain = self.gamma * np.sum(
                weights[np.newaxis, :] * u[np.newaxis, :] * (1.0 - beta),
                axis=1,
            )
            marginal = p + continuation_gain

            best = int(np.argmax(marginal))
            immediate_value += float(p[best])
            u *= beta[best]
            allocation[best] += 1

        value = immediate_value + self.gamma * float(np.sum(weights * (1.0 - u)))
        return allocation, value

    def plan(self, state: RecruitingState) -> GFPPlanResult:
        covariates = np.asarray(state.frontier_covariates, dtype=np.float64)
        max_s = self._candidate_budget_limit(state)
        candidate_values = np.full(max_s + 1, -np.inf, dtype=np.float64)
        allocations: list[np.ndarray] = []

        alpha = self.laplace_provider.alpha(covariates) if covariates.shape[0] else np.empty(
            (0, self.value_surrogate.latent_dim),
            dtype=np.float64,
        )

        for s in range(max_s + 1):
            alloc_s, value_s = self._greedy_for_budget(
                covariates=covariates,
                round_budget=s,
                remaining_after_spend=state.budget_remaining - s,
                alpha=alpha,
            )
            allocations.append(alloc_s)
            candidate_values[s] = value_s

        best_budget = int(np.argmax(candidate_values))
        return GFPPlanResult(
            budget=best_budget,
            allocation=allocations[best_budget],
            value=float(candidate_values[best_budget]),
            candidate_values=candidate_values,
        )

    def act(self, state: RecruitingState) -> np.ndarray:
        return self.plan(state).allocation
