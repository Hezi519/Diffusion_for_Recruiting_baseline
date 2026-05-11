"""
Synthetic covariate transition model for workshop experiments.

Generates child covariates from parent covariates using a categorical
inheritance kernel: each covariate group is inherited from the parent
with a per-group probability, otherwise drawn uniformly at random.

By default uses EMPIRICAL_INHERIT_PROBS derived from 73,669 ICPSR 22140
recruiter-recruit pairs across all disease subnetworks. Pass a scalar
inherit_prob to override all groups with a single value (e.g. for ablations).
"""

from __future__ import annotations

import pickle

import numpy as np
from torch.utils.data import Dataset

from src.data.covariate_spec import COVARIATE_DIM, COVARIATE_GROUPS, EMPIRICAL_INHERIT_PROBS
from src.models.covariate_model.abstract_covariate_model import AbstractCovariateModel


class SyntheticCovariateModel(AbstractCovariateModel):
    """Categorical inheritance covariate transition kernel.

    For each categorical group in the 72-dim one-hot schema, a child
    inherits the parent's value with a per-group probability. Otherwise
    the group is sampled uniformly at random.

    Args:
        inherit_prob: If None (default), uses EMPIRICAL_INHERIT_PROBS derived
            from ICPSR 22140 dyad data (all networks). If a float, overrides all groups uniformly.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        inherit_prob: float | None = None,
        seed: int = 42,
    ) -> None:
        self.inherit_prob = inherit_prob
        self.seed = seed
        if inherit_prob is None:
            self._group_probs = {name: EMPIRICAL_INHERIT_PROBS[name] for name, _, _ in COVARIATE_GROUPS}
        else:
            self._group_probs = {name: inherit_prob for name, _, _ in COVARIATE_GROUPS}

    def train(self, dataset: Dataset, **kwargs) -> dict:
        return {"final_loss": 0.0}

    def sample(
        self,
        parent_covariates: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate one child covariate per parent row.

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

        for name, start, end in COVARIATE_GROUPS:
            group_size = end - start
            p = self._group_probs[name]
            parent_active = np.argmax(parent_covariates[:, start:end], axis=1)
            inherit_mask = rng.random(n) < p
            random_choices = rng.integers(0, group_size, size=n)
            chosen = np.where(inherit_mask, parent_active, random_choices)
            children[np.arange(n), start + chosen] = 1

        return children

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"inherit_prob": self.inherit_prob, "seed": self.seed}, f)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> SyntheticCovariateModel:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(inherit_prob=data["inherit_prob"], seed=data["seed"])
