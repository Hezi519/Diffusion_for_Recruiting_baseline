# Budget DQN with Greedy Allocation

This module implements a **value-based reinforcement learning baseline** for budget allocation. The key idea is to **learn only the total budget decision using DQN**, while using a fixed **greedy allocation rule** to distribute the budget across nodes.

## 1. Key Idea

This approach **decouples the problem**:

- RL learns:

  $$
  b_t = \text{how much budget to spend}
  $$

- Greedy allocator decides:
  $$
  \mathbf{a}_t = \text{where to spend it}
  $$

So the policy becomes:

$$
\pi(s_t) = \text{GreedyAllocator}(s_t, b_t)
$$

---

## 2. State Representation

We use a **DeepSets-style encoder** inside the Q-network:

- Node embedding:

  $$
  h_i = \phi(x_i)
  $$

- Mean pooling:

  $$
  h_{\text{pool}} = \frac{1}{n_t} \sum_i h_i
  $$

- Final input:
  $$
  s_t = [h_{\text{pool}},\ n_t,\ B_t,\ t]
  $$

This allows handling **variable-size frontiers**.

---

## 3. Q-Network

The Q-network predicts values over possible budgets:

$$
Q(s_t, b) \quad \text{for } b \in \{0, \dots, B_{\max}\}
$$

Invalid actions (i.e., $b > B_t$) are masked during selection.

---

## 4. Greedy Allocation Rule

Given a selected budget $b_t$, allocation is constructed sequentially.

### Step-wise allocation:

For each unit of budget:

1. Compute current expected recruits:

   $$
   \text{counts}_{\text{now}} = \text{count\_model}(x, a)
   $$

2. For each node $i$, evaluate marginal gain:

   $$
   \Delta_i = \text{count\_model}(x, a + e_i)[i] - \text{counts}_{\text{now}}[i]
   $$

3. Select:

   $$
   i^* = \arg\max_i \Delta_i
   $$

4. Update:
   $$
   a_{i^*} \leftarrow a_{i^*} + 1
   $$

This repeats $b_t$ times.

---

## 5. Policy

The policy is:

- **Epsilon-greedy** over Q-values for budget selection
- **Deterministic greedy allocator** for node-level allocation

---

## 6. Training Objective

Standard DQN temporal-difference target:

$$
y = r_t + \gamma \max_{b'} Q(s_{t+1}, b')
$$

Loss:

$$
\mathcal{L} = \big(Q(s_t, b_t) - y\big)^2
$$

---

## 7. Replay Buffer

Transitions stored:

$$
(s_t, b_t, r_t, s_{t+1}, \text{done})
$$

Unlike standard DQN:

- States are **variable-size objects**
- Stored as `RecruitingState` instead of fixed tensors

---

## 8. Training Loop

At each step:

1. Select budget $b_t$ via epsilon-greedy
2. Convert to allocation using greedy rule
3. Step environment
4. Store transition
5. Sample minibatch
6. Compute TD loss
7. Update Q-network
8. Soft-update target network
