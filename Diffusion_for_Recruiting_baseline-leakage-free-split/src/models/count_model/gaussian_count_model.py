"""
Gaussian Process Regression count model.

Uses sklearn GPR to predict covariates -> out-degree (number of children
recruited). Following the reference explanation, this corresponds to
covariates -> (degree - 1) where degree is the undirected degree of a node
and -1 removes the parent-edge so only children are counted.

Post-inference: round to integer, floor at 0, cap at allocation. This
rounding step distorts the effective probability mass distribution — a known
limitation. Consider the DDPM-based count model for a distributional approach.

References:
    https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
"""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from src.models.count_model.abstract_count_model import AbstractCountModel


class GaussianCountModel(AbstractCountModel):
    """GPR-based count model.

    Predicts covariates -> out-degree (number of children a frontier member
    will recruit). In the paper's notation this is (degree - 1) where degree
    is the total undirected degree and -1 removes the incoming parent edge.

    Key design choices vs. a naive GPR:
    - ``normalize_y=True``: shifts the prior mean to the empirical training
      mean, which is essential because count targets are non-negative and
      far from zero.
    - ``alpha``: small observation-noise regularisation for numerical
      stability and to reflect that integer degree observations carry noise.
    - ``max_train_size``: GPR fit is O(n³) — subsampling protects against
      OOM / timeout on large graphs while retaining representativeness.

    Args:
        seed: Random seed for the GPR and subsampling.
        alpha: Noise regularisation added to the diagonal of the kernel
            matrix (sklearn ``alpha`` parameter). Default 0.1 works well
            for integer count targets.
        n_restarts_optimizer: Number of kernel hyper-parameter restarts.
            Higher values reduce risk of poor local optima but increase
            fit time.
        max_train_size: If the training set has more rows than this, a
            random subsample of this size is used. None means no limit.
            Recommended: 2000 for most machines.
    """

    def __init__(
        self,
        seed: int = 42,
        alpha: float = 0.1,
        n_restarts_optimizer: int = 5,
        max_train_size: Optional[int] = 2000,
    ) -> None:
        self.seed = seed
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_train_size = max_train_size

        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
        )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        self._fitted = False

    def fit(
        self,
        covariates: np.ndarray,
        degrees: np.ndarray,
    ) -> None:
        """Fit the GPR on covariates -> out-degree.

        Args:
            covariates: (N, D) training covariate vectors (typically D=72).
            degrees: (N,) observed out-degrees, i.e. number of children
                recruited by each node. Must be non-negative integers.
        """
        covariates = np.asarray(covariates, dtype=np.float64)
        degrees = np.asarray(degrees, dtype=np.float64)

        if covariates.ndim != 2:
            raise ValueError(f"covariates must be 2-D, got shape {covariates.shape}")
        if degrees.ndim != 1:
            raise ValueError(f"degrees must be 1-D, got shape {degrees.shape}")
        if covariates.shape[0] != degrees.shape[0]:
            raise ValueError(
                f"covariates ({covariates.shape[0]}) and degrees ({degrees.shape[0]}) "
                "must have the same number of rows"
            )
        if np.any(degrees < 0):
            raise ValueError("degrees must be non-negative")

        # Subsample for GPR scalability (O(n^3) fit)
        n = covariates.shape[0]
        if self.max_train_size is not None and n > self.max_train_size:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n, size=self.max_train_size, replace=False)
            covariates = covariates[idx]
            degrees = degrees[idx]

        self.gpr.fit(covariates, degrees)
        self._fitted = True

    def predict(
        self,
        frontier_covariates: np.ndarray,
        allocations: np.ndarray,
    ) -> np.ndarray:
        """Predict recruit counts, clipped to [0, allocation].

        Post-inference processing: round to nearest integer, clip to [0,
        allocation]. This is the acknowledged limitation of a regression
        approach — it distorts the probability mass.

        Args:
            frontier_covariates: (n_t, D) covariate vectors.
            allocations: (n_t,) budget allocated per individual.

        Returns:
            (n_t,) non-negative integer counts, each <= corresponding allocation.
        """
        if not self._fitted:
            raise RuntimeError("GaussianCountModel.fit() must be called before predict()")

        frontier_covariates = np.asarray(frontier_covariates, dtype=np.float64)
        allocations = np.asarray(allocations, dtype=int)

        if frontier_covariates.ndim != 2:
            raise ValueError(
                f"frontier_covariates must be 2-D, got shape {frontier_covariates.shape}"
            )
        if allocations.shape != (frontier_covariates.shape[0],):
            raise ValueError(
                f"allocations shape {allocations.shape} must match "
                f"(n_t,) = ({frontier_covariates.shape[0]},)"
            )

        raw = self.gpr.predict(frontier_covariates)
        counts = np.round(raw).astype(int)
        counts = np.clip(counts, 0, allocations)
        return counts

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "gpr": self.gpr,
                    "seed": self.seed,
                    "alpha": self.alpha,
                    "n_restarts_optimizer": self.n_restarts_optimizer,
                    "max_train_size": self.max_train_size,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> GaussianCountModel:
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(
            seed=data["seed"],
            alpha=data.get("alpha", 0.1),
            n_restarts_optimizer=data.get("n_restarts_optimizer", 5),
            max_train_size=data.get("max_train_size", 2000),
        )
        model.gpr = data["gpr"]
        model._fitted = True
        return model
