from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random

from src.environment.state import RecruitingState


@dataclass
class Batch:
    obs: List[RecruitingState]
    action: List[int]
    reward: List[float]
    next_obs: List[RecruitingState]
    done: List[bool]


class ReplayBuffer:
    """
    Replay buffer for variable-size RecruitingState objects.

    Stores transitions:
        (state, action, reward, next_state, done)

    Unlike a standard DQN replay buffer, this version does NOT assume
    observations are fixed-size numpy arrays. It is designed for the
    recruiting environment, where each state's frontier_covariates has
    variable shape (n_t, 72).
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)

        self.obs_buf: List[RecruitingState] = []
        self.act_buf: List[int] = []
        self.rew_buf: List[float] = []
        self.next_obs_buf: List[RecruitingState] = []
        self.done_buf: List[bool] = []

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: RecruitingState,
        action: int,
        reward: float,
        next_obs: RecruitingState,
        done: bool,
    ) -> None:
        """
        Add one transition to the replay buffer.
        """
        if self.size < self.capacity:
            self.obs_buf.append(obs)
            self.act_buf.append(int(action))
            self.rew_buf.append(float(reward))
            self.next_obs_buf.append(next_obs)
            self.done_buf.append(bool(done))
            self.size += 1
        else:
            self.obs_buf[self.ptr] = obs
            self.act_buf[self.ptr] = int(action)
            self.rew_buf[self.ptr] = float(reward)
            self.next_obs_buf[self.ptr] = next_obs
            self.done_buf[self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> Batch:
        """
        Sample a batch of transitions uniformly at random.
        """
        assert self.size > 0, "ReplayBuffer is empty"

        if self.size < batch_size:
            idxs = random.choices(range(self.size), k=batch_size)
        else:
            idxs = random.sample(range(self.size), k=batch_size)

        return Batch(
            obs=[self.obs_buf[i] for i in idxs],
            action=[self.act_buf[i] for i in idxs],
            reward=[self.rew_buf[i] for i in idxs],
            next_obs=[self.next_obs_buf[i] for i in idxs],
            done=[self.done_buf[i] for i in idxs],
        )

    @property
    def size_filled(self) -> int:
        return self.size