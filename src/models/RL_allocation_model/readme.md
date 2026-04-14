# RL-based Structured Budget Allocation

This module implements a **value-based reinforcement learning framework for structured budget allocation** over a dynamically evolving frontier. The core idea is to decompose a complex combinatorial action into a sequence of structured decisions, while still learning long-term value through temporal-difference learning.

Instead of directly learning over a combinatorial action space, we factorize the decision into three components:

1. **Budget selection**

   $$
   b_t \sim Q_{\text{budget}}(s_t)
   $$

2. **Support size (top-k selection)**

   $
   k_t \sim Q_k(s_t)
   $

3. **Node-level scoring**
   $$
   s_i = Q_{\text{node}}(s_t, x_i)
   $$

These are combined into a final allocation using a deterministic builder.

---

## 1. State Representation

The state is encoded using a **DeepSets-style encoder**:

- Node embedding:

  $$
  h_i = \phi(x_i)
  $$

- Mean pooling:

  $$
  h_{\text{pool}} = \frac{1}{n_t} \sum_{i=1}^{n_t} h_i
  $$

- Final state vector:
  $$
  s_t = \big[h_{\text{pool}},\ n_t,\ B_t,\ t\big]
  $$

This produces a fixed-dimensional representation regardless of frontier size.

---

## 2. Three-Head Q Network

The model outputs three components:

### 2.1 Budget Head

$$
Q_{\text{budget}}(s_t) \in \mathbb{R}^{B_{\max} + 1}
$$

### 2.2 k Head

$$
Q_k(s_t) \in \mathbb{R}^{K_{\max} + 1}
$$

### 2.3 Node Score Head

$$
Q_{\text{node}}(s_t, x_i)
$$

Each node score is computed by concatenating the global state with node features.

---

## 3. Allocation Construction

Given $b_t$, $k_t$, and node scores:

1. Select top-$k$ nodes by score
2. Apply softmax within selected nodes:

   $$
   w_i = \frac{\exp(s_i)}{\sum_{j \in \text{top-}k} \exp(s_j)}
   $$

3. Allocate budget proportionally:

   $$
   \tilde{a}_i = b_t \cdot w_i
   $$

4. Convert to integers using **largest-remainder rounding**

This guarantees:

- Exact total budget
- At most $k$ active nodes
- Smooth dependence on learned scores

---

## 4. Policy

The policy is **value-based with structured decoding**:

- Budget and $k$: epsilon-greedy over Q-values
- Node scores: deterministic (with optional noise during training)
- Allocation: deterministic builder

Overall:

$$
\pi(s_t) = \text{Builder}(b_t, k_t, Q_{\text{node}})
$$

---

## 5. Training Objective

### 5.1 Budget Head (TD Learning)

$$
y^{(b)} = r_t + \gamma \max_{b'} Q_{\text{budget}}(s_{t+1}, b')
$$

### 5.2 k Head (TD Learning)

$$
y^{(k)} = r_t + \gamma \max_{k'} Q_k(s_{t+1}, k')
$$

---

## 6. Node Score Learning (Key Idea)

Node scores represent **marginal long-term value per unit budget**:

$$
Q_{\text{node}}(s, i) \approx \text{value of allocating 1 unit to node } i
$$

### Target:

$$
y_i =
\underbrace{\text{immediate gain}_i}_{\text{count model}}
+
\gamma \cdot
\underbrace{\max_b Q_{\text{budget}}(s_i', b)}_{\text{future value}}
$$

Where:

- $s_i'$ is the next state after allocating 1 unit to node $i$
- Immediate gain is computed via the count model

This enables **TD-style learning without multi-step rollout**.

---

## 7. Replay Buffer

We store transitions:

$$
(s_t, b_t, k_t, r_t, s_{t+1}, \text{done})
$$

Node-level targets are recomputed during training.

---

## 8. Target Network

A separate target network is used for stability:

- Periodically updated
- Used for TD target computation

---

## 9. Training Loop

At each step:

1. Act using current policy
2. Step environment
3. Store transition
4. Sample minibatch from replay buffer
5. Compute:
   - budget loss
   - k loss
   - node loss
6. Update network parameters
7. Periodically update target network
