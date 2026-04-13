from __future__ import annotations

from dataclasses import dataclass
from typing import List
import copy
import numpy as np

from src.environment.state import RecruitingState


@dataclass
class Transition:
    state: RecruitingState
    action_budget: int
    action_k: int
    reward: float
    next_state: RecruitingState
    done: bool


class ReplayBuffer:
    """
    Simple uniform replay buffer for structured allocation RL.

    Stores transitions at the decision level:
        (state, chosen budget, chosen k, reward, next_state, done)

    Note:
        We do NOT store node_scores/allocation here because they can be
        recomputed from the current online network during training.
    """

    def __init__(self, capacity: int = 10000, seed: int = 42) -> None:
        self.capacity = int(capacity)
        self.buffer: List[Transition] = []
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def add(self, transition: Transition) -> None:
        t = Transition(
            state=RecruitingState(
                frontier_covariates=np.array(transition.state.frontier_covariates, copy=True),
                budget_remaining=int(transition.state.budget_remaining),
                timestep=int(transition.state.timestep),
            ),
            action_budget=int(transition.action_budget),
            action_k=int(transition.action_k),
            reward=float(transition.reward),
            next_state=RecruitingState(
                frontier_covariates=np.array(transition.next_state.frontier_covariates, copy=True),
                budget_remaining=int(transition.next_state.budget_remaining),
                timestep=int(transition.next_state.timestep),
            ),
            done=bool(transition.done),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self.ptr] = t

        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Requested batch_size={batch_size}, but buffer only has {len(self.buffer)} samples."
            )
        idx = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)