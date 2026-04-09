from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.environment.state import RecruitingState
from src.models.RL_allocation_model.allocation_builder import build_allocation
from src.models.RL_allocation_model.q_network import QOutput


@dataclass
class ValuePolicyStep:
    state_vec: torch.Tensor
    budget: int
    k: int
    node_scores: torch.Tensor
    allocation: np.ndarray
    budget_q: torch.Tensor
    k_q: torch.Tensor


class StructuredValuePolicy:
    """
    Value-based structured policy:
        state -> budget_q, k_q, node_scores
        choose budget and k with epsilon-greedy
        build allocation with top-k + soft scores
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        q_network: torch.nn.Module,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        self.encoder = encoder
        self.q_network = q_network
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        self.encoder.to(self.device)
        self.q_network.to(self.device)

    def train(self) -> None:
        self.encoder.train()
        self.q_network.train()

    def eval(self) -> None:
        self.encoder.eval()
        self.q_network.eval()

    def _state_to_tensors(
        self,
        state: RecruitingState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frontier_covariates = torch.tensor(
            state.frontier_covariates,
            dtype=torch.float32,
            device=self.device,
        )
        budget_remaining = torch.tensor(
            float(state.budget_remaining),
            dtype=torch.float32,
            device=self.device,
        )
        timestep = torch.tensor(
            float(state.timestep),
            dtype=torch.float32,
            device=self.device,
        )
        return frontier_covariates, budget_remaining, timestep

    def encode_state(
        self,
        state: RecruitingState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frontier_covariates, budget_remaining, timestep = self._state_to_tensors(state)

        state_vec = self.encoder(
            frontier_covariates=frontier_covariates,
            budget_remaining=budget_remaining,
            timestep=timestep,
        )
        return state_vec, frontier_covariates

    def _select_budget(
        self,
        budget_q: torch.Tensor,
        budget_remaining: int,
        epsilon: float,
    ) -> int:
        masked_q = self.q_network.masked_budget_q(
            budget_q=budget_q,
            budget_remaining=budget_remaining,
        )
        max_allowed = min(self.q_network.max_budget, max(0, int(budget_remaining)))

        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, max_allowed + 1))
        return int(torch.argmax(masked_q).item())

    def _select_k(
        self,
        k_q: torch.Tensor,
        frontier_size: int,
        chosen_budget: int,
        epsilon: float,
    ) -> int:
        max_allowed = min(
            self.q_network.max_k,
            max(0, int(frontier_size)),
            max(0, int(chosen_budget)),
        )

        masked_q = self.q_network.masked_k_q(
            k_q=k_q,
            max_allowed=max_allowed,
        )

        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, max_allowed + 1))
        return int(torch.argmax(masked_q).item())

    def _build_step(
        self,
        state_vec: torch.Tensor,
        q_out: QOutput,
        budget: int,
        k: int,
        score_noise_std: float = 0.0,
    ) -> ValuePolicyStep:
        scores = q_out.node_scores
        if score_noise_std > 0.0 and scores.numel() > 0:
            noise = torch.randn_like(scores) * score_noise_std
            scores_for_builder = scores + noise
        else:
            scores_for_builder = scores

        allocation = build_allocation(
            budget=budget,
            k=k,
            scores=scores_for_builder.detach().cpu().numpy(),
        )

        return ValuePolicyStep(
            state_vec=state_vec,
            budget=budget,
            k=k,
            node_scores=q_out.node_scores,
            allocation=allocation,
            budget_q=q_out.budget_q,
            k_q=q_out.k_q,
        )

    def act(
        self,
        state: RecruitingState,
        epsilon_budget: float = 0.1,
        epsilon_k: float = 0.1,
        score_noise_std: float = 0.1,
    ) -> ValuePolicyStep:
        state_vec, frontier_covariates = self.encode_state(state)

        q_out = self.q_network(
            state_vec=state_vec,
            frontier_covariates=frontier_covariates,
        )

        budget = self._select_budget(
            budget_q=q_out.budget_q,
            budget_remaining=state.budget_remaining,
            epsilon=epsilon_budget,
        )
        k = self._select_k(
            k_q=q_out.k_q,
            frontier_size=state.frontier_size,
            chosen_budget=budget,
            epsilon=epsilon_k,
        )

        return self._build_step(
            state_vec=state_vec,
            q_out=q_out,
            budget=budget,
            k=k,
            score_noise_std=score_noise_std,
        )

    @torch.no_grad()
    def act_greedy(
        self,
        state: RecruitingState,
    ) -> ValuePolicyStep:
        state_vec, frontier_covariates = self.encode_state(state)

        q_out = self.q_network(
            state_vec=state_vec,
            frontier_covariates=frontier_covariates,
        )

        budget_q = self.q_network.masked_budget_q(q_out.budget_q, state.budget_remaining)
        budget = int(torch.argmax(budget_q).item())

        max_allowed_k = min(
            self.q_network.max_k,
            state.frontier_size,
            budget,
        )
        k_q = self.q_network.masked_k_q(q_out.k_q, max_allowed=max_allowed_k)
        k = int(torch.argmax(k_q).item())

        return self._build_step(
            state_vec=state_vec,
            q_out=q_out,
            budget=budget,
            k=k,
            score_noise_std=0.0,
        )