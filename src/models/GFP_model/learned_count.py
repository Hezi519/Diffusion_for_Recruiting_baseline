from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CensoredCountFitConfig:
    n_samples: int = 2048
    max_allocation: int = 10
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-3


class _PoissonRateNet(nn.Module):
    def __init__(self, covariate_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x)).squeeze(-1) + 1e-6


class LearnedCensoredPoissonSurvival:
    """
    Learned q_psi(c | x) for GFP via censored Poisson likelihood.

    Training data are triples (x, k, y), where y = min(C, k). If y < k,
    the likelihood uses P(C = y | x); if y == k, it uses P(C >= k | x).
    """

    def __init__(
        self,
        covariate_dim: int = 72,
        hidden_dim: int = 64,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.model = _PoissonRateNet(covariate_dim, hidden_dim).to(self.device)
        self._rng = np.random.default_rng(seed + 1)

    @staticmethod
    def _poisson_cdf_less_than(rate: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        max_k = int(torch.max(k).detach().cpu().item()) if k.numel() else 0
        cdf = torch.zeros_like(rate)
        pmf = torch.exp(-rate)
        cdf = torch.where(k > 0, pmf, cdf)
        for c in range(1, max_k):
            pmf = pmf * rate / float(c)
            cdf = torch.where(k > c, cdf + pmf, cdf)
        return torch.clamp(cdf, 0.0, 1.0)

    def _nll(
        self,
        x: torch.Tensor,
        allocation: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        rate = self.model(x)
        log_pmf = observed * torch.log(rate) - rate - torch.lgamma(observed + 1.0)
        cdf_less_than_k = self._poisson_cdf_less_than(rate, allocation.long())
        log_survival = torch.log(torch.clamp(1.0 - cdf_less_than_k, min=1e-8))
        is_censored = observed.long() >= allocation.long()
        log_likelihood = torch.where(is_censored, log_survival, log_pmf)
        return -log_likelihood.mean()

    def fit(
        self,
        covariates: np.ndarray,
        allocations: np.ndarray,
        observed_counts: np.ndarray,
        epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> dict:
        x = torch.tensor(covariates, dtype=torch.float32, device=self.device)
        k = torch.tensor(allocations, dtype=torch.float32, device=self.device)
        y = torch.tensor(observed_counts, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        n = x.shape[0]
        losses: list[float] = []

        for _ in range(max(1, epochs)):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, max(1, batch_size)):
                idx = perm[start:start + batch_size]
                loss = self._nll(x[idx], k[idx], y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

        return {
            "final_loss": losses[-1] if losses else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "n_samples": int(n),
        }

    @torch.no_grad()
    def rates(self, covariates: np.ndarray) -> np.ndarray:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates[np.newaxis, :]
        x = torch.tensor(covariates, dtype=torch.float32, device=self.device)
        return self.model(x).detach().cpu().numpy().astype(np.float64)

    def predict(
        self,
        frontier_covariates: np.ndarray,
        allocations: np.ndarray,
    ) -> np.ndarray:
        """Sample censored recruit counts for use inside RecruitingEnv."""
        frontier_covariates = np.asarray(frontier_covariates, dtype=np.float64)
        allocations = np.asarray(allocations, dtype=int)
        rates = self.rates(frontier_covariates)
        counts = self._rng.poisson(rates).astype(int)
        return np.clip(counts, 0, allocations)

    def survival_prob(
        self,
        covariates: np.ndarray,
        ell: int | np.ndarray,
    ) -> np.ndarray:
        rates = self.rates(covariates)
        ell_arr = np.asarray(ell, dtype=int)
        if ell_arr.ndim == 0:
            ell_arr = np.full(rates.shape, int(ell_arr), dtype=int)
        else:
            ell_arr = np.broadcast_to(ell_arr, rates.shape).astype(int)

        out = np.ones_like(rates, dtype=np.float64)
        for idx, k in enumerate(ell_arr):
            if k <= 0:
                continue
            lam = float(rates[idx])
            pmf = np.exp(-lam)
            cdf = pmf
            for c in range(1, int(k)):
                pmf *= lam / float(c)
                cdf += pmf
            out[idx] = float(np.clip(1.0 - cdf, 0.0, 1.0))
        return out

    def tau(
        self,
        covariates: np.ndarray,
        alpha: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        covariates = np.asarray(covariates, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        k = np.asarray(k, dtype=int)
        n, latent_dim = alpha.shape
        tau = np.ones((n, latent_dim), dtype=np.float64)
        rates = self.rates(covariates)

        for i in range(n):
            ki = int(k[i])
            if ki <= 0:
                continue
            lam = float(rates[i])
            pmf = np.exp(-lam)
            vals = np.full(latent_dim, pmf, dtype=np.float64)
            cdf = pmf
            alpha_power = np.ones(latent_dim, dtype=np.float64)
            for c in range(1, ki):
                pmf *= lam / float(c)
                cdf += pmf
                alpha_power *= alpha[i]
                vals += pmf * alpha_power
            vals += max(0.0, 1.0 - cdf) * (alpha[i] ** ki)
            tau[i] = np.clip(vals, 1e-12, 1.0)

        return tau


def make_censored_count_dataset(
    base_count_model,
    covariate_pool: np.ndarray,
    cfg: CensoredCountFitConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    covariate_pool = np.asarray(covariate_pool, dtype=np.float64)
    parent_idx = rng.integers(0, covariate_pool.shape[0], size=cfg.n_samples)
    covariates = covariate_pool[parent_idx]
    allocations = rng.integers(1, max(2, cfg.max_allocation + 1), size=cfg.n_samples)
    observed = base_count_model.predict(covariates, allocations)
    return covariates, allocations.astype(int), np.asarray(observed, dtype=int)
