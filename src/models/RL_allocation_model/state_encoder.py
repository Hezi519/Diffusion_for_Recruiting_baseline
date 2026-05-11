from __future__ import annotations

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, covariate_dim: int = 72, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.phi = nn.Sequential(
            nn.Linear(covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # pooled embedding + frontier_size + budget_remaining + timestep
        self.output_dim = hidden_dim + 3

    def forward(
        self,
        frontier_covariates: torch.Tensor,
        budget_remaining: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if frontier_covariates.shape[0] == 0:
            pooled = torch.zeros(self.hidden_dim, device=frontier_covariates.device)
            frontier_size = torch.tensor([0.0], device=frontier_covariates.device)
        else:
            h = self.phi(frontier_covariates)
            pooled = h.mean(dim=0)
            frontier_size = torch.tensor(
                [float(frontier_covariates.shape[0])],
                device=frontier_covariates.device,
            )

        if budget_remaining.ndim == 0:
            budget_remaining = budget_remaining.unsqueeze(0)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        state_vec = torch.cat(
            [pooled, frontier_size, budget_remaining, timestep],
            dim=0,
        )
        return state_vec