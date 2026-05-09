from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.covariate_spec import COVARIATE_DIM, COVARIATE_GROUPS


@dataclass
class OffspringFitConfig:
    n_pairs: int = 4096
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-3


class _GroupedCategoricalNet(nn.Module):
    def __init__(self, covariate_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, end - start) for _, start, end in COVARIATE_GROUPS]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.trunk(x)
        return [head(h) for head in self.heads]


class LearnedCategoricalOffspringModel:
    """
    Learned G_theta(x' | x) for one-hot tabular covariates.

    The conditional distribution factorizes across the existing categorical
    groups, producing valid one-hot child covariates on sample().
    """

    def __init__(
        self,
        covariate_dim: int = COVARIATE_DIM,
        hidden_dim: int = 64,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.model = _GroupedCategoricalNet(covariate_dim, hidden_dim).to(self.device)

    @staticmethod
    def _targets(children: np.ndarray) -> list[np.ndarray]:
        targets = []
        for _, start, end in COVARIATE_GROUPS:
            targets.append(np.argmax(children[:, start:end], axis=1).astype(np.int64))
        return targets

    def fit(
        self,
        parent_covariates: np.ndarray,
        child_covariates: np.ndarray,
        epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> dict:
        parents = torch.tensor(parent_covariates, dtype=torch.float32, device=self.device)
        targets = [
            torch.tensor(t, dtype=torch.long, device=self.device)
            for t in self._targets(np.asarray(child_covariates))
        ]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        n = parents.shape[0]
        losses: list[float] = []

        for _ in range(max(1, epochs)):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, max(1, batch_size)):
                idx = perm[start:start + batch_size]
                logits = self.model(parents[idx])
                loss = torch.stack(
                    [F.cross_entropy(logit, target[idx]) for logit, target in zip(logits, targets)]
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

        return {
            "final_loss": losses[-1] if losses else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "n_pairs": int(n),
        }

    @torch.no_grad()
    def sample(
        self,
        parent_covariates: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        parent_covariates = np.asarray(parent_covariates, dtype=np.float64)
        if parent_covariates.ndim == 1:
            parent_covariates = parent_covariates[np.newaxis, :]

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        parents = torch.tensor(parent_covariates, dtype=torch.float32, device=self.device)
        logits_by_group = self.model(parents)

        children = np.zeros((parent_covariates.shape[0], COVARIATE_DIM), dtype=int)
        for logits, (_, start, end) in zip(logits_by_group, COVARIATE_GROUPS):
            probs = torch.softmax(logits, dim=1)
            sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
            children[np.arange(children.shape[0]), start + sampled.cpu().numpy()] = 1
        return children


def make_offspring_dataset(
    base_covariate_model,
    covariate_pool: np.ndarray,
    cfg: OffspringFitConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    covariate_pool = np.asarray(covariate_pool, dtype=np.float64)
    parent_idx = rng.integers(0, covariate_pool.shape[0], size=cfg.n_pairs)
    parents = covariate_pool[parent_idx]
    children = base_covariate_model.sample(parents, seed=int(rng.integers(1 << 31)))
    return parents, children
