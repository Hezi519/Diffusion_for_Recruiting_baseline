"""
Gaussian Process Regression count model (reference/stub implementation).

Uses sklearn GPR to predict covariates -> degree, as described in
reference/diffusion/explanation.md. Post-inference: round to integer,
floor at 0, cap at allocation.

This is a simple baseline. The count model person may replace this
with a more sophisticated model.
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from src.models.count_model.abstract_count_model import AbstractCountModel


class GaussianCountModel(AbstractCountModel):
    """GPR-based count model.

    Trains covariates -> (degree - 1) following the approach described
    in the reference explanation.md. The -1 accounts for the parent
    edge (a node's out-degree in the directed graph minus the edge
    from its own recruiter).

    Args:
        seed: Random seed for the GPR.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            random_state=seed,
            n_restarts_optimizer=2,
        )
        self._fitted = False

    def fit(
        self,
        covariates: np.ndarray,
        degrees: np.ndarray,
    ) -> None:
        """Fit the GPR on covariates -> degree.

        Args:
            covariates: (N, 72) training covariate vectors.
            degrees: (N,) observed out-degrees (number of children recruited).
        """
        self.gpr.fit(covariates, degrees)
        self._fitted = True

    def predict(
        self,
        frontier_covariates: np.ndarray,
        allocations: np.ndarray,
    ) -> np.ndarray:
        """Predict counts, clipped to [0, allocation].

        Args:
            frontier_covariates: (n_t, 72) covariate vectors.
            allocations: (n_t,) budget per individual.

        Returns:
            (n_t,) non-negative integer counts, each <= corresponding allocation.
        """
        if not self._fitted:
            raise RuntimeError("GaussianCountModel.fit() must be called before predict()")

        raw = self.gpr.predict(frontier_covariates)
        counts = np.round(raw).astype(int)
        counts = np.clip(counts, 0, allocations.astype(int))
        return counts

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"gpr": self.gpr, "seed": self.seed}, f)

    @classmethod
    def load(cls, path: str) -> GaussianCountModel:
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(seed=data["seed"])
        model.gpr = data["gpr"]
        model._fitted = True
        return model
