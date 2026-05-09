from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class QOutput:
    budget_q: torch.Tensor      # (max_budget + 1,)
    k_q: torch.Tensor           # (max_k + 1,)
    node_scores: torch.Tensor   # (n_t,)


class ThreeHeadQNetwork(nn.Module):
    """
    Value-based 3-head network:
        1. budget Q-values
        2. k Q-values
        3. node-level marginal Q-like scores
    """

    def __init__(
        self,
        state_dim: int,
        covariate_dim: int = 72,
        hidden_dim: int = 128,
        max_budget: int = 20,
        max_k: int = 20,
    ) -> None:
        super().__init__()

        self.state_dim = int(state_dim)
        self.covariate_dim = int(covariate_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_budget = int(max_budget)
        self.max_k = int(max_k)

        self.budget_head = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.max_budget + 1),
        )

        self.k_head = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.max_k + 1),
        )

        self.node_score_head = nn.Sequential(
            nn.Linear(self.state_dim + self.covariate_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        state_vec: torch.Tensor,
        frontier_covariates: torch.Tensor,
    ) -> QOutput:
        if state_vec.ndim != 1:
            raise ValueError(f"state_vec must be 1-D, got shape {tuple(state_vec.shape)}")

        if frontier_covariates.ndim != 2:
            raise ValueError(
                f"frontier_covariates must be 2-D, got shape {tuple(frontier_covariates.shape)}"
            )

        if frontier_covariates.shape[1] != self.covariate_dim:
            raise ValueError(
                f"frontier_covariates second dim must be {self.covariate_dim}, "
                f"got {frontier_covariates.shape[1]}"
            )

        budget_q = self.budget_head(state_vec)
        k_q = self.k_head(state_vec)

        n_t = frontier_covariates.shape[0]
        if n_t == 0:
            node_scores = torch.empty(
                0,
                dtype=state_vec.dtype,
                device=state_vec.device,
            )
        else:
            repeated_state = state_vec.unsqueeze(0).expand(n_t, -1)
            node_inputs = torch.cat([repeated_state, frontier_covariates], dim=1)
            node_scores = self.node_score_head(node_inputs).squeeze(-1)

        return QOutput(
            budget_q=budget_q,
            k_q=k_q,
            node_scores=node_scores,
        )

    def masked_budget_q(
        self,
        budget_q: torch.Tensor,
        budget_remaining: int,
    ) -> torch.Tensor:
        if budget_q.ndim != 1 or budget_q.shape[0] != self.max_budget + 1:
            raise ValueError(
                f"budget_q must have shape ({self.max_budget + 1},), got {tuple(budget_q.shape)}"
            )

        max_allowed = min(self.max_budget, max(0, int(budget_remaining)))
        masked = budget_q.clone()
        if max_allowed + 1 < masked.shape[0]:
            masked[max_allowed + 1:] = -1e9
        return masked

    def masked_k_q(
        self,
        k_q: torch.Tensor,
        max_allowed: int,
    ) -> torch.Tensor:
        if k_q.ndim != 1 or k_q.shape[0] != self.max_k + 1:
            raise ValueError(
                f"k_q must have shape ({self.max_k + 1},), got {tuple(k_q.shape)}"
            )

        max_allowed = min(self.max_k, max(0, int(max_allowed)))
        masked = k_q.clone()
        if max_allowed + 1 < masked.shape[0]:
            masked[max_allowed + 1:] = -1e9
        return masked