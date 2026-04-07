"""Abstract policy interface for the recruiting environment."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.environment.state import RecruitingState


class AbstractPolicy(ABC):
    """Interface for recruitment allocation policies.

    Any policy (RL, heuristic, random) implements act() to return
    a per-frontier-member allocation vector.
    """

    @abstractmethod
    def act(self, state: RecruitingState) -> np.ndarray:
        """Choose an allocation for the current state.

        Args:
            state: Current recruiting state with frontier covariates and budget.

        Returns:
            (n_t,) array of non-negative integer allocations.
            Must satisfy: sum(result) <= state.budget_remaining.
        """
