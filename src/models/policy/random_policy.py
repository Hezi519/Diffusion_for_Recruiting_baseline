"""Random allocation policy."""

from __future__ import annotations

import numpy as np

from src.environment.state import RecruitingState


class RandomPolicy:
    """Randomly allocate budget across frontier members.

    Each round, spends up to budget_remaining by distributing
    units one-at-a-time to random frontier members.

    Args:
        seed: Random seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, state: RecruitingState) -> np.ndarray:
        n = state.frontier_size
        budget = state.budget_remaining

        if n == 0 or budget <= 0:
            return np.zeros(n, dtype=int)

        max_per_person = max(1, budget // n)
        action = self.rng.integers(0, max_per_person + 1, size=n)

        total = action.sum()
        if total > budget:
            action = np.zeros(n, dtype=int)
            for _ in range(budget):
                action[self.rng.integers(0, n)] += 1

        return action
