from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrontierValueSurrogate(nn.Module):
    """
    Budget-conditioned exponential-saturation frontier value surrogate.

        V(r, F) = sum_j w_j(r) * (1 - exp(-sum_{x in F} h_j(x)))

    h(x) is constrained nonnegative through softplus, and w(r) is nonnegative
    with sum_j w_j(r) = r for r > 0.
    """

    def __init__(
        self,
        covariate_dim: int = 72,
        latent_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.covariate_dim = int(covariate_dim)
        self.latent_dim = int(latent_dim)

        self.coverage_net = nn.Sequential(
            nn.Linear(self.covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.weight_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def coverage(self, covariates: torch.Tensor) -> torch.Tensor:
        if covariates.ndim != 2:
            raise ValueError(f"covariates must be 2-D, got {tuple(covariates.shape)}")
        return F.softplus(self.coverage_net(covariates))

    def weights(self, budget_remaining: torch.Tensor | int | float) -> torch.Tensor:
        device = next(self.parameters()).device
        if not torch.is_tensor(budget_remaining):
            budget_remaining = torch.tensor(
                float(budget_remaining),
                dtype=torch.float32,
                device=device,
            )
        else:
            budget_remaining = budget_remaining.to(device=device, dtype=torch.float32)
        if budget_remaining.ndim == 0:
            budget_remaining = budget_remaining.unsqueeze(0)

        r = budget_remaining
        logits = self.weight_net(r.reshape(1, 1)).squeeze(0)
        weights = torch.clamp(r.reshape(()), min=0.0) * torch.softmax(logits, dim=0)
        if float(r.reshape(()).detach().cpu()) <= 0.0:
            return torch.zeros_like(weights)
        return weights

    def forward(
        self,
        frontier_covariates: torch.Tensor,
        budget_remaining: torch.Tensor | int | float,
    ) -> torch.Tensor:
        if frontier_covariates.shape[0] == 0:
            z = torch.zeros(
                self.latent_dim,
                dtype=frontier_covariates.dtype,
                device=frontier_covariates.device,
            )
        else:
            z = self.coverage(frontier_covariates).sum(dim=0)
        w = self.weights(budget_remaining).to(frontier_covariates.device)
        return torch.sum(w * (1.0 - torch.exp(-z)))
