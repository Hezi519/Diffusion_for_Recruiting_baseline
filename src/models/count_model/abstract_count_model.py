"""
Abstract interface for the count model p_phi(m_t | X_t, a_t).

The count model predicts how many new recruits each frontier member
produces given their covariates and the allocation they receive.
The count model person should subclass this and implement predict().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractCountModel(ABC):
    """Interface for count models.

    Given frontier covariates and per-individual allocations, predicts
    how many new recruits each frontier member produces.
    """

    @abstractmethod
    def predict(
        self,
        frontier_covariates: np.ndarray,
        allocations: np.ndarray,
    ) -> np.ndarray:
        """Predict number of new recruits per frontier member.

        Args:
            frontier_covariates: (n_t, 72) covariate vectors for frontier individuals.
            allocations: (n_t,) budget allocated to each individual (non-negative ints).

        Returns:
            counts: (n_t,) array of non-negative integers. counts[i] is the number
                    of children predicted for frontier member i. Should satisfy
                    counts[i] <= allocations[i].
        """

    def save(self, path: str) -> None:
        """Save model to disk. Optional — override if needed."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> AbstractCountModel:
        """Load model from disk. Optional — override if needed."""
        raise NotImplementedError
