"""
State representation for the recruiting MDP.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RecruitingState:
    """State s_t = (X_t, b_t) in the recruiting MDP.

    Attributes:
        frontier_covariates: (n_t, 72) covariate vectors for current frontier.
                             n_t varies across timesteps.
        budget_remaining: Non-negative integer, remaining budget.
        timestep: Current round number (0-indexed).
    """

    frontier_covariates: np.ndarray  # (n_t, 72)
    budget_remaining: int
    timestep: int = 0

    @property
    def frontier_size(self) -> int:
        return self.frontier_covariates.shape[0]

    def to_dict(self) -> dict:
        """Serialize state for logging/debugging."""
        return {
            "frontier_covariates": self.frontier_covariates.copy(),
            "budget_remaining": self.budget_remaining,
            "frontier_size": self.frontier_size,
            "timestep": self.timestep,
        }
