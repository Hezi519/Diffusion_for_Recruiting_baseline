"""
Gym-like environment for multi-round recruitment resource allocation.

Combines a covariate generation model and a count model to simulate
the MDP defined in diffusion_for_recruiting.md:

    s_t = (X_t, b_t)
    a_t = allocation vector over frontier
    m_i ~ count_model(x_i, a_i)  for each frontier member i
    children_i ~ covariate_model(x_i) repeated m_i times
    X_{t+1} = union of all children
    b_{t+1} = b_t - sum(a_t)
    r_t = total new recruits
"""

from __future__ import annotations

import numpy as np

from src.data.covariate_spec import COVARIATE_DIM
from src.environment.state import RecruitingState
from src.models.count_model.abstract_count_model import AbstractCountModel
from src.models.covariate_model.abstract_covariate_model import AbstractCovariateModel


class RecruitingEnv:
    """Multi-round recruiting environment.

    Args:
        covariate_model: Generates child covariates from parent covariates.
        count_model: Predicts number of children per frontier member.
        initial_budget: Total budget for the episode.
        discount_factor: Discount factor gamma for reward computation.
        max_rounds: Maximum number of rounds before forced termination.
        seed: Random seed for stochastic operations.
    """

    def __init__(
        self,
        covariate_model: AbstractCovariateModel,
        count_model: AbstractCountModel,
        initial_budget: int,
        discount_factor: float = 0.9,
        max_rounds: int = 50,
        seed: int = 42,
    ) -> None:
        self.covariate_model = covariate_model
        self.count_model = count_model
        self.initial_budget = initial_budget
        self.discount_factor = discount_factor
        self.max_rounds = max_rounds
        self.rng = np.random.default_rng(seed)
        self._seed = seed

        self._state: RecruitingState | None = None
        self._episode_rewards: list[float] = []
        self._done: bool = True

    def reset(
        self,
        initial_frontier: np.ndarray,
        seed: int | None = None,
    ) -> RecruitingState:
        """Start a new episode.

        Args:
            initial_frontier: (n_0, 72) covariate vectors for the initial frontier.
            seed: Optional seed for this episode.

        Returns:
            Initial state s_0.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        initial_frontier = np.asarray(initial_frontier, dtype=np.float64)
        assert initial_frontier.ndim == 2 and initial_frontier.shape[1] == COVARIATE_DIM

        self._state = RecruitingState(
            frontier_covariates=initial_frontier,
            budget_remaining=self.initial_budget,
            timestep=0,
        )
        self._episode_rewards = []
        self._done = False
        return self._state

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[RecruitingState, float, bool, dict]:
        """Execute one round of the recruiting MDP.

        Args:
            action: (n_t,) array of non-negative integer allocations.
                    Must satisfy: sum(action) <= budget_remaining.

        Returns:
            next_state: RecruitingState with new frontier and updated budget.
            reward: Immediate reward (number of new recruits).
            done: Whether the episode has terminated.
            info: Dict with diagnostic information.
        """
        assert not self._done, "Episode is done. Call reset()."
        action = np.asarray(action, dtype=int)
        assert action.shape == (self._state.frontier_size,), \
            f"Action shape {action.shape} != frontier size ({self._state.frontier_size},)"
        assert np.all(action >= 0), "Allocations must be non-negative."

        budget_spent = int(action.sum())
        assert budget_spent <= self._state.budget_remaining, \
            f"Action spends {budget_spent} but only {self._state.budget_remaining} remaining"

        # Step 1: Count model predicts per-parent recruit counts
        counts = self.count_model.predict(
            self._state.frontier_covariates, action
        )
        counts = np.asarray(counts, dtype=int)
        assert np.all(counts >= 0) and np.all(counts <= action), \
            "Count model returned counts outside [0, allocation]"

        # Step 2: Generate child covariates for each parent
        new_covariates = []
        for i in range(self._state.frontier_size):
            m_i = int(counts[i])
            if m_i > 0:
                parent_cov = np.tile(
                    self._state.frontier_covariates[i], (m_i, 1)
                )
                children = self.covariate_model.sample(
                    parent_cov,
                    seed=int(self.rng.integers(1 << 31)),
                )
                new_covariates.append(children)

        # Step 3: Assemble new frontier
        if new_covariates:
            next_frontier = np.concatenate(new_covariates, axis=0)
        else:
            next_frontier = np.empty((0, COVARIATE_DIM))

        # Step 4: Compute reward
        total_recruits = next_frontier.shape[0]
        reward = float(total_recruits)

        # Step 5: Update budget
        new_budget = self._state.budget_remaining - budget_spent

        # Step 6: Build next state
        next_state = RecruitingState(
            frontier_covariates=next_frontier,
            budget_remaining=new_budget,
            timestep=self._state.timestep + 1,
        )

        # Step 7: Check termination
        termination_reason = None
        if new_budget <= 0:
            termination_reason = "budget_exhausted"
        elif next_frontier.shape[0] == 0:
            termination_reason = "empty_frontier"
        elif next_state.timestep >= self.max_rounds:
            termination_reason = "max_rounds"

        done = termination_reason is not None

        # Step 8: Update internal state
        self._state = next_state
        self._done = done
        self._episode_rewards.append(reward)

        info = {
            "counts": counts,
            "total_recruits": total_recruits,
            "budget_spent": budget_spent,
            "round": next_state.timestep,
            "termination_reason": termination_reason,
        }

        return next_state, reward, done, info

    @property
    def state(self) -> RecruitingState:
        """Current state."""
        assert self._state is not None, "Call reset() first."
        return self._state

    @property
    def cumulative_reward(self) -> float:
        """Total undiscounted reward accumulated in this episode."""
        return sum(self._episode_rewards)

    @property
    def cumulative_discounted_reward(self) -> float:
        """Total discounted reward accumulated in this episode."""
        return sum(
            r * (self.discount_factor ** t)
            for t, r in enumerate(self._episode_rewards)
        )
