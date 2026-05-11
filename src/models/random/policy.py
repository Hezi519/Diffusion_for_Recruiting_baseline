"""Random allocation policy."""

from __future__ import annotations

import numpy as np

from src.environment.state import RecruitingState


class RandomPolicy:
    """Uniformly pick a round-budget, then multinomial-drop it.

    Each round:
      s ~ Uniform{0, 1, ..., budget_remaining}
      action ~ Multinomial(s, uniform over n frontier members)

    Knob-free null baseline: both the schedule (how much to spend this
    round) and the split (who gets it) are uniform random.

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

        s = int(self.rng.integers(0, budget + 1))
        return self.rng.multinomial(s, np.full(n, 1.0 / n)).astype(int)
