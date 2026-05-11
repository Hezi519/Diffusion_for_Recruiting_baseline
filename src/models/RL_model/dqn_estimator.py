from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.RL_model.replay_buffer import ReplayBuffer
from src.models.RL_model.greedy_allocator import greedy_allocator as default_greedy_allocator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    tau: float = 0.01
    batch_size: int = 64
    buffer_capacity: int = 100_000
    warmup_steps: int = 500
    train_episodes: int = 300
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    target_update_interval: int = 1
    hidden_dim: int = 128
    max_grad_norm: float = 10.0
    covariate_dim: int = 72


class SetBudgetQNet(nn.Module):
    """
    Input:
        frontier_covariates: list of [n_i, d] tensors
        budget_remaining: [B]
        timestep: [B]
    Output:
        Q-values [B, num_actions]
    """

    def __init__(self, covariate_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.phi = nn.Sequential(
            nn.Linear(covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # pooled + frontier_size + budget + timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(
        self,
        frontier_covariates: List[torch.Tensor],
        budget_remaining: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        pooled_list = []

        for x in frontier_covariates:
            if x.shape[0] == 0:
                pooled = torch.zeros(self.hidden_dim, dtype=torch.float32, device=x.device)
                frontier_size = torch.tensor([0.0], dtype=torch.float32, device=x.device)
            else:
                h = self.phi(x)
                pooled = h.mean(dim=0)
                frontier_size = torch.tensor([float(x.shape[0])], dtype=torch.float32, device=x.device)

            pooled_list.append(torch.cat([pooled, frontier_size], dim=0))

        pooled_batch = torch.stack(pooled_list, dim=0)
        extra = torch.stack([budget_remaining, timestep], dim=1)
        joint = torch.cat([pooled_batch, extra], dim=1)

        return self.rho(joint)


class BudgetDQNSolver:
    """
    DQN learns a scalar action:
        spend_budget in {0, 1, ..., initial_budget}

    Then converts spend_budget -> allocation vector using a fixed greedy rule.
    """

    def __init__(
        self,
        env,
        initial_frontier_fn: Callable[[], np.ndarray],
        budget_allocator: Callable[[object, int], np.ndarray],
        cfg: DQNConfig,
        seed: int = 42,
    ):
        self.env = env
        self.initial_frontier_fn = initial_frontier_fn
        self.budget_allocator = budget_allocator
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.max_budget = int(env.initial_budget)
        self.num_actions = self.max_budget + 1

        self.policy_net = SetBudgetQNet(
            covariate_dim=cfg.covariate_dim,
            num_actions=self.num_actions,
            hidden_dim=cfg.hidden_dim,
        ).to(DEVICE)

        self.target_net = SetBudgetQNet(
            covariate_dim=cfg.covariate_dim,
            num_actions=self.num_actions,
            hidden_dim=cfg.hidden_dim,
        ).to(DEVICE)

        self.target_net.load_state_dict(copy.deepcopy(self.policy_net.state_dict()))
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(cfg.buffer_capacity)

        self.epsilon = cfg.eps_start
        self.total_steps = 0
        self.loss_history: List[float] = []

    def _reset_state(self, seed: int | None = None):
        initial_frontier = self.initial_frontier_fn()
        if seed is None:
            seed = int(self.rng.integers(1 << 31))
        return self.env.reset(initial_frontier, seed=seed)

    def _allowed_actions(self, state) -> List[int]:
        rb = int(state.budget_remaining)
        return list(range(rb + 1))

    def _single_q_values(self, net, state) -> np.ndarray:
        frontier = torch.tensor(
            state.frontier_covariates,
            dtype=torch.float32,
            device=DEVICE,
        )
        budget = torch.tensor(
            [float(state.budget_remaining)],
            dtype=torch.float32,
            device=DEVICE,
        )
        timestep = torch.tensor(
            [float(state.timestep)],
            dtype=torch.float32,
            device=DEVICE,
        )

        with torch.no_grad():
            qvals = net([frontier], budget, timestep)[0].detach().cpu().numpy()

        return qvals

    def _batch_q_values(self, net, states) -> torch.Tensor:
        frontiers = [
            torch.tensor(s.frontier_covariates, dtype=torch.float32, device=DEVICE)
            for s in states
        ]
        budgets = torch.tensor(
            [float(s.budget_remaining) for s in states],
            dtype=torch.float32,
            device=DEVICE,
        )
        timesteps = torch.tensor(
            [float(s.timestep) for s in states],
            dtype=torch.float32,
            device=DEVICE,
        )
        return net(frontiers, budgets, timesteps)

    def select_action(self, state, greedy: bool = False) -> int:
        allowed = self._allowed_actions(state)
        assert len(allowed) > 0, "No valid actions available"

        if (not greedy) and (random.random() < self.epsilon):
            return random.choice(allowed)

        qvals = self._single_q_values(self.policy_net, state)

        masked = np.full_like(qvals, -1e9, dtype=np.float32)
        masked[allowed] = qvals[allowed]
        return int(np.argmax(masked))

    def _env_step_with_budget(self, state, spend_budget: int):
        alloc_vec = self.budget_allocator(state, spend_budget)
        next_state, reward, done, info = self.env.step(alloc_vec)
        return next_state, float(reward), bool(done), info

    def _update_one_step(self):
        if self.buffer.size_filled < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return

        batch = self.buffer.sample(self.cfg.batch_size)

        actions = torch.tensor(batch.action, dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)

        q_values = self._batch_q_values(self.policy_net, batch.obs)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_all = self._batch_q_values(self.target_net, batch.next_obs)

            for i, s in enumerate(batch.next_obs):
                rb = int(s.budget_remaining)
                if rb + 1 < self.num_actions:
                    next_q_all[i, rb + 1:] = -1e9

            next_q = next_q_all.max(dim=1).values
            target = rewards + (1.0 - dones) * self.cfg.gamma * next_q

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.loss_history.append(float(loss.detach().cpu()))

    def _soft_update_target(self):
        tau = self.cfg.tau
        with torch.no_grad():
            for p_t, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
                p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def train_one_episode(self):
        state = self._reset_state()

        while True:
            action_budget = self.select_action(state, greedy=False)
            next_state, reward, done, _ = self._env_step_with_budget(state, action_budget)

            self.buffer.add(state, action_budget, reward, next_state, done)
            self._update_one_step()

            state = next_state
            self.total_steps += 1

            if done:
                break

        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)

    def save_checkpoint(self, path: str, episode: int) -> None:
        """Save training state so training can be resumed if interrupted."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "episode": episode,
            "loss_history": self.loss_history,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load training state. Returns the episode to resume from."""
        ckpt = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.total_steps = ckpt["total_steps"]
        self.loss_history = ckpt["loss_history"]
        return int(ckpt["episode"])

    def save_weights(self, path: str) -> None:
        """Save final policy network weights only."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def evaluate(self, n_episodes_eval: int, seed_offset: int = 200000) -> np.ndarray:
        episode_rewards: List[float] = []

        for ep in range(n_episodes_eval):
            state = self._reset_state(seed=seed_offset + ep)
            total_reward = 0.0

            while True:
                action_budget = self.select_action(state, greedy=True)
                next_state, reward, done, _ = self._env_step_with_budget(state, action_budget)

                total_reward += float(reward)
                state = next_state

                if done:
                    break

            episode_rewards.append(total_reward)

        return np.asarray(episode_rewards, dtype=np.float32)


def run_budget_dqn(
    env,
    initial_frontier_fn: Callable[[], np.ndarray],
    budget_allocator: Callable[[object, int], np.ndarray] | None,
    n_episodes_eval: int,
    seed: int,
    cfg: DQNConfig,
    log_every_n_episodes: int = 10,
    on_new_best: Callable[["BudgetDQNSolver", int, float], None] | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
    weights_path: str | None = None,
):
    import os
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if budget_allocator is None:
        budget_allocator = lambda state, k: default_greedy_allocator(state, k, env.count_model)

    learner = BudgetDQNSolver(
        env=env,
        initial_frontier_fn=initial_frontier_fn,
        budget_allocator=budget_allocator,
        cfg=cfg,
        seed=seed,
    )

    # Resume from checkpoint if one exists
    start_episode = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_episode = learner.load_checkpoint(checkpoint_path)
        print(f"Resumed from checkpoint at episode {start_episode}/{cfg.train_episodes}")

    best_eval_reward = -1.0
    best_eval_episode = -1
    history: List[dict] = []

    for ep in range(start_episode, cfg.train_episodes):
        learner.train_one_episode()

        if (ep + 1) % learner.cfg.target_update_interval == 0:
            learner._soft_update_target()

        if (ep + 1) % log_every_n_episodes == 0:
            rewards_eval = learner.evaluate(n_episodes_eval=max(1, n_episodes_eval // 2))
            mean_eval_reward = float(np.mean(rewards_eval)) if rewards_eval.size > 0 else 0.0
            recent_loss = float(np.mean(learner.loss_history[-50:])) if learner.loss_history else 0.0

            history.append({
                "episode": ep + 1,
                "eval_reward": mean_eval_reward,
                "epsilon": learner.epsilon,
                "total_steps": learner.total_steps,
                "recent_loss": recent_loss,
            })

            print(
                f"[DQN {ep+1:>4}/{cfg.train_episodes}]  "
                f"eval={mean_eval_reward:6.1f}  "
                f"loss={recent_loss:.4f}  "
                f"eps={learner.epsilon:.3f}  "
                f"steps={learner.total_steps}"
            )

            if mean_eval_reward > best_eval_reward:
                best_eval_reward = mean_eval_reward
                best_eval_episode = ep + 1
                if on_new_best is not None:
                    on_new_best(learner, ep + 1, mean_eval_reward)

        if checkpoint_path and (ep + 1) % checkpoint_every == 0:
            learner.save_checkpoint(checkpoint_path, ep + 1)
            print(f"  [checkpoint saved at ep {ep+1}]")

    rewards = learner.evaluate(n_episodes_eval=n_episodes_eval)

    if weights_path:
        learner.save_weights(weights_path)
        print(f"Model weights saved to: {weights_path}")

    if best_eval_reward < 0:
        best_eval_reward = 0.0
        best_eval_episode = 0

    return rewards, learner, best_eval_reward, best_eval_episode, history