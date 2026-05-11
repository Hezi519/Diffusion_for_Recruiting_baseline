"""
Synthetic covariate transition model for workshop experiments.

Generates child covariates from parent covariates using a simple
categorical inheritance kernel: each covariate group is inherited
from the parent with probability `inherit_prob`, otherwise drawn
uniformly at random from that group.

This creates the covariate-dependent offspring structure that
Generative Frontier Planning (GFP) is designed to exploit:
high-rate parents generate high-rate children, so allocating
budget to high-rate frontier nodes compounds over rounds.

No training is required — the model is an oracle parametric kernel.
"""

from __future__ import annotations

import pickle

import numpy as np
from torch.utils.data import Dataset

from src.data.covariate_spec import COVARIATE_DIM, COVARIATE_GROUPS
from src.models.covariate_model.abstract_covariate_model import AbstractCovariateModel


class SyntheticCovariateModel(AbstractCovariateModel):
    """Categorical inheritance covariate transition kernel.

    For each categorical group in the 72-dim one-hot schema, a child
    inherits the parent's value with probability `inherit_prob`.
    Otherwise the group is sampled uniformly at random.

    Args:
        inherit_prob: Per-group inheritance probability (0 = fully random,
            1 = exact copy of parent).  Default 0.7 gives children that
            are clearly related to their parent while retaining diversity.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        inherit_prob: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.inherit_prob = inherit_prob
        self.seed = seed

    def train(
        self,
        dataset: Dataset,
        epochs: int = 0,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 42,
        log_interval: int = 100,
    ) -> dict:
        """No-op: this model requires no training."""
        return {"final_loss": 0.0}

    def sample(
        self,
        parent_covariates: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate one child covariate per parent row.

        For each covariate group: with probability `inherit_prob` the child
        copies the parent's active category; otherwise a random category in
        that group is chosen.

        Args:
            parent_covariates: (n, 72) one-hot parent covariate vectors.
            seed: Random seed for this call.

        Returns:
            (n, 72) valid one-hot child covariate vectors.
        """
        parent_covariates = np.asarray(parent_covariates, dtype=np.float64)
        if parent_covariates.ndim == 1:
            parent_covariates = parent_covariates[np.newaxis, :]

        rng = np.random.default_rng(seed)
        n = parent_covariates.shape[0]
        children = np.zeros((n, COVARIATE_DIM), dtype=int)

        for _, start, end in COVARIATE_GROUPS:
            group_size = end - start
            parent_active = np.argmax(parent_covariates[:, start:end], axis=1)
            inherit_mask = rng.random(n) < self.inherit_prob
            random_choices = rng.integers(0, group_size, size=n)
            chosen = np.where(inherit_mask, parent_active, random_choices)
            children[np.arange(n), start + chosen] = 1

        return children

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"inherit_prob": self.inherit_prob, "seed": self.seed},
                f,
            )

    @classmethod
    def load(cls, path: str, device: str = "auto") -> SyntheticCovariateModel:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(inherit_prob=data["inherit_prob"], seed=data["seed"])
