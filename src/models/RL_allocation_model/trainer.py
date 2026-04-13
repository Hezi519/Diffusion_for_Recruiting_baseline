from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import copy
from src.models.RL_allocation_model.replay_buffer import ReplayBuffer, Transition
from src.environment.state import RecruitingState
from src.models.RL_allocation_model.policy import StructuredValuePolicy


@dataclass
class ValueTrainerConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    train_episodes: int = 300
    max_steps_per_episode: int = 100

    epsilon_budget_start: float = 0.20
    epsilon_budget_end: float = 0.05

    epsilon_k_start: float = 0.20
    epsilon_k_end: float = 0.05

    score_noise_start: float = 0.20
    score_noise_end: float = 0.02

    node_score_loss_weight: float = 1.0

    # replay buffer
    replay_buffer_capacity: int = 20000
    batch_size: int = 32
    min_buffer_size: int = 128
    updates_per_env_step: int = 1

    # target network
    target_update_interval: int = 200

    # multi-sample averaging for node TD target
    node_target_num_samples: int = 1


class StructuredQTrainer:
    """
    Value-based trainer.

    - budget head: TD target
    - k head: TD target
    - node score head: node-level TD-like target
    - replay buffer
    - target network
    - optional multi-sample averaging for node targets
    """

    def __init__(
        self,
        env,
        policy: StructuredValuePolicy,
        initial_frontier_fn,
        count_model,
        covariate_model,
        cfg: ValueTrainerConfig,
        device: str = "cpu",
        seed: int = 42,
        on_new_best: Callable[[StructuredValuePolicy, int, float], None] | None = None,
        n_episodes_eval: int = 10,
        log_every_n_episodes: int = 10,
    ):
        self.env = env
        self.policy = policy
        self.initial_frontier_fn = initial_frontier_fn
        self.count_model = count_model
        self.covariate_model = covariate_model
        self.cfg = cfg
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.on_new_best = on_new_best
        self.n_episodes_eval = n_episodes_eval
        self.log_every_n_episodes = log_every_n_episodes

        self.optimizer = torch.optim.Adam(
            list(self.policy.encoder.parameters()) + list(self.policy.q_network.parameters()),
            lr=cfg.lr,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=cfg.replay_buffer_capacity,
            seed=seed,
        )

        self.target_encoder = copy.deepcopy(self.policy.encoder).to(self.device)
        self.target_q_network = copy.deepcopy(self.policy.q_network).to(self.device)

        self.target_encoder.eval()
        self.target_q_network.eval()

        self.global_step = 0
        self._update_target_network()

    def _update_target_network(self) -> None:
        self.target_encoder.load_state_dict(self.policy.encoder.state_dict())
        self.target_q_network.load_state_dict(self.policy.q_network.state_dict())
        self.target_encoder.eval()
        self.target_q_network.eval()
        
    def _linear_schedule(self, start: float, end: float, episode_idx: int) -> float:
        if self.cfg.train_episodes <= 1:
            return end
        frac = episode_idx / float(self.cfg.train_episodes - 1)
        return float(start + frac * (end - start))

    def _masked_max_budget_q(
        self,
        network,
        q_values: torch.Tensor,
        budget_remaining: int,
    ) -> torch.Tensor:
        masked = network.masked_budget_q(q_values, budget_remaining)
        return masked.max()

    def _masked_max_k_q(
        self,
        network,
        q_values: torch.Tensor,
        frontier_size: int,
        budget_limit: int,
    ) -> torch.Tensor:
        max_allowed = min(
            network.max_k,
            max(0, int(frontier_size)),
            max(0, int(budget_limit)),
        )
        masked = network.masked_k_q(q_values, max_allowed=max_allowed)
        return masked.max()

    @torch.no_grad()
    def _encode_counterfactual_state_target(
        self,
        state: RecruitingState,
        next_frontier: np.ndarray,
        new_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = RecruitingState(
            frontier_covariates=next_frontier,
            budget_remaining=new_budget,
            timestep=state.timestep + 1,
        )
        return self._encode_state_target(next_state)


    def _encode_state_online(
        self,
        state: RecruitingState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.encode_state(state)
    
    @torch.no_grad()
    def _encode_state_target(
        self,
        state: RecruitingState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frontier_covariates = torch.tensor(
            state.frontier_covariates,
            dtype=torch.float32,
            device=self.device,
        )
        budget_remaining = torch.tensor(
            float(state.budget_remaining),
            dtype=torch.float32,
            device=self.device,
        )
        timestep = torch.tensor(
            float(state.timestep),
            dtype=torch.float32,
            device=self.device,
        )

        state_vec = self.target_encoder(
            frontier_covariates=frontier_covariates,
            budget_remaining=budget_remaining,
            timestep=timestep,
        )
        return state_vec, frontier_covariates

    def _batched_counterfactual_transitions(
        self,
        state: RecruitingState,
    ) -> list[tuple[float, np.ndarray, int]]:
        """
        For every node i in the frontier, simulate the one-step transition
        where we allocate exactly 1 unit to node i. Returns a list of
        (immediate_gain_i, next_frontier_i, new_budget_i), one per node.

        Single batched diffusion call across all nodes, instead of n separate
        calls. Still supports averaging over node_target_num_samples stochastic
        samples per node.
        """
        n = state.frontier_size
        cov_dim = state.frontier_covariates.shape[1]
        new_budget = state.budget_remaining - 1
        num_samples = max(1, int(self.cfg.node_target_num_samples))

        # Per-node child counts under "1 unit to node i": since only node i has
        # non-zero allocation, count_model.predict({1 at i, 0 elsewhere}) gives
        # counts[j] = 0 for j != i. Under that allocation-clipping contract, the
        # child count for node i is the same as what count_model.predict would
        # return at that single node with allocation=1. Compute it n times.
        per_node_counts = np.zeros(n, dtype=int)
        for i in range(n):
            alloc_i = np.zeros(n, dtype=int)
            alloc_i[i] = 1
            counts_i = self.count_model.predict(state.frontier_covariates, alloc_i)
            per_node_counts[i] = int(counts_i[i])

        gains_per_node = np.zeros(n, dtype=np.float64)
        last_frontiers: list[np.ndarray] = [
            np.empty((0, cov_dim), dtype=np.float64) for _ in range(n)
        ]

        for _ in range(num_samples):
            parent_rows = []
            owner_node = []
            for i in range(n):
                m_i = int(per_node_counts[i])
                if m_i > 0:
                    parent_rows.append(
                        np.tile(state.frontier_covariates[i], (m_i, 1))
                    )
                    owner_node.extend([i] * m_i)

            if parent_rows:
                all_parents = np.concatenate(parent_rows, axis=0)
                all_children = self.covariate_model.sample(
                    all_parents,
                    seed=int(self.rng.integers(1 << 31)),
                )
                owner_node_arr = np.asarray(owner_node, dtype=int)
                for i in range(n):
                    mask = owner_node_arr == i
                    frontier_i = all_children[mask] if mask.any() else np.empty(
                        (0, cov_dim), dtype=np.float64
                    )
                    gains_per_node[i] += float(frontier_i.shape[0])
                    last_frontiers[i] = frontier_i
            # nodes with m_i == 0 stay at gain 0 and empty frontier

        gains_per_node /= float(num_samples)

        return [
            (float(gains_per_node[i]), last_frontiers[i], new_budget)
            for i in range(n)
        ]

    def _compute_loss_on_transition(
        self,
        transition: Transition,
    ) -> torch.Tensor:
        state = transition.state
        next_state = transition.next_state
        reward = transition.reward
        done = transition.done
        action_budget = transition.action_budget
        action_k = transition.action_k

        state_vec, frontier_covariates = self._encode_state_online(state)
        q_out = self.policy.q_network(
            state_vec=state_vec,
            frontier_covariates=frontier_covariates,
        )

        with torch.no_grad():
            next_state_vec, next_frontier_covariates = self._encode_state_target(next_state)
            next_q_out = self.target_q_network(
                state_vec=next_state_vec,
                frontier_covariates=next_frontier_covariates,
            )

            reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)

            if done:
                budget_target = reward_t
                k_target = reward_t
            else:
                budget_target = reward_t + self.cfg.gamma * self._masked_max_budget_q(
                    self.target_q_network,
                    next_q_out.budget_q,
                    next_state.budget_remaining,
                )
                k_target = reward_t + self.cfg.gamma * self._masked_max_k_q(
                    self.target_q_network,
                    next_q_out.k_q,
                    next_state.frontier_size,
                    next_state.budget_remaining,
                )

        pred_budget_q = q_out.budget_q[action_budget]
        pred_k_q = q_out.k_q[action_k]

        budget_loss = F.mse_loss(pred_budget_q, budget_target)
        k_loss = F.mse_loss(pred_k_q, k_target)

        node_targets = self._compute_node_td_targets(state)
        if node_targets.numel() == 0:
            node_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            node_loss = F.mse_loss(q_out.node_scores, node_targets)

        loss = budget_loss + k_loss + self.cfg.node_score_loss_weight * node_loss
        return loss
    
    def _train_on_batch(
        self,
        batch: list[Transition],
    ) -> float:
        losses = []

        for transition in batch:
            loss = self._compute_loss_on_transition(transition)
            losses.append(loss)

        batch_loss = torch.stack(losses).mean()

        self.optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.encoder.parameters()) + list(self.policy.q_network.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()

        return float(batch_loss.item())
    
    def _compute_node_td_targets(
        self,
        state: RecruitingState,
    ) -> torch.Tensor:
        """
        For each node i:
            target_i = immediate_gain_i + gamma * max_b Q_budget(s_i', b)
        where s_i' is the counterfactual next state after allocating 1 unit to node i.
        """
        n = state.frontier_size
        if n == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        targets = []

        with torch.no_grad():
            counterfactuals = self._batched_counterfactual_transitions(state)

            for immediate_gain_i, next_frontier_i, new_budget_i in counterfactuals:
                next_state_vec_i, next_frontier_tensor_i = self._encode_counterfactual_state_target(
                    state=state,
                    next_frontier=next_frontier_i,
                    new_budget=new_budget_i,
                )

                next_q_out_i = self.target_q_network(
                    state_vec=next_state_vec_i,
                    frontier_covariates=next_frontier_tensor_i,
                )

                if new_budget_i <= 0:
                    future_value_i = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                else:
                    future_value_i = self._masked_max_budget_q(
                        self.target_q_network,
                        next_q_out_i.budget_q,
                        new_budget_i,
                    )

                target_i = torch.tensor(
                    immediate_gain_i,
                    dtype=torch.float32,
                    device=self.device,
                ) + self.cfg.gamma * future_value_i

                targets.append(target_i)

        return torch.stack(targets, dim=0)

    def train_one_episode(
        self,
        episode_idx: int,
    ):
        epsilon_budget = self._linear_schedule(
            self.cfg.epsilon_budget_start,
            self.cfg.epsilon_budget_end,
            episode_idx,
        )
        epsilon_k = self._linear_schedule(
            self.cfg.epsilon_k_start,
            self.cfg.epsilon_k_end,
            episode_idx,
        )
        score_noise_std = self._linear_schedule(
            self.cfg.score_noise_start,
            self.cfg.score_noise_end,
            episode_idx,
        )

        state = self.env.reset(self.initial_frontier_fn())
        self.policy.train()

        total_reward = 0.0
        total_loss = 0.0
        num_updates = 0
        step_count = 0

        while True:
            # 1) act in environment using online policy
            step = self.policy.act(
                state,
                epsilon_budget=epsilon_budget,
                epsilon_k=epsilon_k,
                score_noise_std=score_noise_std,
            )

            next_state, reward, done, _ = self.env.step(step.allocation)
            total_reward += reward

            # 2) store transition
            transition = Transition(
                state=state,
                action_budget=step.budget,
                action_k=step.k,
                reward=float(reward),
                next_state=next_state,
                done=bool(done),
            )
            self.replay_buffer.add(transition)

            # 3) train from replay buffer
            if len(self.replay_buffer) >= self.cfg.min_buffer_size:
                for _ in range(self.cfg.updates_per_env_step):
                    batch = self.replay_buffer.sample(self.cfg.batch_size)
                    batch_loss = self._train_on_batch(batch)
                    total_loss += batch_loss
                    num_updates += 1

                    self.global_step += 1
                    if self.global_step % self.cfg.target_update_interval == 0:
                        self._update_target_network()

            step_count += 1
            state = next_state

            if done or step_count >= self.cfg.max_steps_per_episode:
                break

        return {
            "episode_return": float(total_reward),
            "avg_loss": float(total_loss / max(num_updates, 1)),
            "episode_len": step_count,
            "epsilon_budget": epsilon_budget,
            "epsilon_k": epsilon_k,
            "score_noise_std": score_noise_std,
            "buffer_size": len(self.replay_buffer),
            "num_updates": num_updates,
        }

    @torch.no_grad()
    def _evaluate_greedy(self, n_episodes: int) -> float:
        """Run greedy rollouts and return mean episode return."""
        self.policy.eval()
        returns = []
        for _ in range(max(1, n_episodes)):
            state = self.env.reset(self.initial_frontier_fn())
            total_reward = 0.0
            step_count = 0
            while True:
                step = self.policy.act_greedy(state)
                state, reward, done, _ = self.env.step(step.allocation)
                total_reward += float(reward)
                step_count += 1
                if done or step_count >= self.cfg.max_steps_per_episode:
                    break
            returns.append(total_reward)
        self.policy.train()
        return float(np.mean(returns)) if returns else 0.0

    def train(self):
        history = []
        best_eval_reward = -1.0
        best_eval_episode = -1

        for ep in range(self.cfg.train_episodes):
            metrics = self.train_one_episode(ep)
            history.append(metrics)

            if (ep + 1) % self.log_every_n_episodes == 0:
                print(
                    f"[Episode {ep+1}] "
                    f"Return={metrics['episode_return']:.2f}, "
                    f"AvgLoss={metrics['avg_loss']:.4f}, "
                    f"Len={metrics['episode_len']}, "
                    f"Buffer={metrics['buffer_size']}, "
                    f"Updates={metrics['num_updates']}"
                )

                mean_eval_reward = self._evaluate_greedy(
                    n_episodes=max(1, self.n_episodes_eval // 2),
                )

                if mean_eval_reward > best_eval_reward:
                    best_eval_reward = mean_eval_reward
                    best_eval_episode = ep + 1
                    if self.on_new_best is not None:
                        self.on_new_best(self.policy, ep + 1, mean_eval_reward)

        return history