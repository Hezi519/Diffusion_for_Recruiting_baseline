"""
Oracle Poisson count model for synthetic experiments.

Ground-truth count distribution:
    C_i ~ Poisson(rate(x_i)),  rate(x_i) = mean_degree * softplus(w^T x_i) / Z

where w is a fixed random weight vector and Z normalises so the expected
rate over the covariate distribution equals mean_degree.  No fitting is
required — the model is ready to use immediately after construction.

Setting heterogeneity > 0 creates a wide spread of high- and low-rate
individuals, making budget allocation non-trivial.
"""

from __future__ import annotations

import pickle

import numpy as np

from src.data.covariate_spec import COVARIATE_DIM, COVARIATE_GROUPS
from src.models.count_model.abstract_count_model import AbstractCountModel


class SyntheticCountModel(AbstractCountModel):
    """Poisson count model with covariate-dependent rates.

    Args:
        mean_degree: Target expected recruit count per unit of budget.
            Controls overall recruitment density.
        heterogeneity: Standard deviation of the weight vector w.
            Higher values create wider spread between high/low-rate nodes.
            Recommended range: 0.5 (mild) to 2.0 (extreme).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        mean_degree: float = 2.5,
        heterogeneity: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.mean_degree = mean_degree
        self.heterogeneity = heterogeneity
        self.seed = seed

        rng = np.random.default_rng(seed)
        self._w = rng.standard_normal(COVARIATE_DIM) * heterogeneity
        self._rng = np.random.default_rng(seed + 1)
        self._scale = self._calibrate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calibrate(self, n_samples: int = 5000) -> float:
        """Compute scale so E[rate(x)] = mean_degree over random covariates."""
        rng = np.random.default_rng(0)
        covs = np.zeros((n_samples, COVARIATE_DIM))
        for row in range(n_samples):
            for _, start, end in COVARIATE_GROUPS:
                covs[row, rng.integers(start, end)] = 1.0
        raw = self._softplus(covs @ self._w)
        mean_raw = raw.mean()
        return self.mean_degree / (mean_raw + 1e-8)

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(np.clip(x, -20.0, 20.0)))

    def _rates(self, covariates: np.ndarray) -> np.ndarray:
        """Return per-node Poisson rates, shape (n,)."""
        return self._softplus(covariates @ self._w) * self._scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def individual_rate(self, covariate: np.ndarray) -> float:
        """Expected recruit rate for one covariate vector (for analysis)."""
        return float(self._rates(covariate[np.newaxis, :])[0])

    def predict(
        self,
        frontier_covariates: np.ndarray,
        allocations: np.ndarray,
    ) -> np.ndarray:
        """Draw recruit counts from Poisson(rate(x_i)), clipped to allocation.

        Args:
            frontier_covariates: (n, D) covariate vectors.
            allocations: (n,) non-negative integer allocations.

        Returns:
            (n,) non-negative integer counts, each <= corresponding allocation.
        """
        frontier_covariates = np.asarray(frontier_covariates, dtype=np.float64)
        allocations = np.asarray(allocations, dtype=int)
        rates = self._rates(frontier_covariates)
        counts = self._rng.poisson(rates).astype(int)
        return np.clip(counts, 0, allocations)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "mean_degree": self.mean_degree,
                    "heterogeneity": self.heterogeneity,
                    "seed": self.seed,
                    "w": self._w,
                    "scale": self._scale,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> SyntheticCountModel:
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(
            mean_degree=data["mean_degree"],
            heterogeneity=data["heterogeneity"],
            seed=data["seed"],
        )
        model._w = data["w"]
        model._scale = data["scale"]
        return model
