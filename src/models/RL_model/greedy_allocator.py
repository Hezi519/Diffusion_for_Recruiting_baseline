import numpy as np


def greedy_allocator(state, spend_budget: int, count_model) -> np.ndarray:
    """
    Greedy allocation based on marginal expected recruits.

    The GPR count model does not use the allocation vector when computing
    predictions — allocation only clips the output at the end:
        predicted[i] = clip(round(gpr(cov_i)), 0, alloc[i])

    This means the marginal gain from adding one unit to node i is:
        1  if round(gpr(cov_i)) > alloc[i]   (one more recruit unlocked)
        0  otherwise                          (already at capacity)

    The optimal greedy strategy is therefore:
      1. Predict each node's uncapped count with one vectorized GPR call.
      2. Give each node up to its predicted count, highest first.
      3. Spread any leftover budget (zero marginal gain) round-robin.

    This is mathematically identical to the original loop but replaces
    O(spend_budget × n_frontier) GPR calls with a single vectorized call —
    a ~500x speedup at budget=500.

    Args:
        state: RecruitingState
        spend_budget: total budget to allocate this round
        count_model: fitted count model (must have predict())

    Returns:
        alloc_vec: (n_t,) integer allocation
    """
    frontier = state.frontier_covariates  # (n, d)
    n = frontier.shape[0]
    alloc = np.zeros(n, dtype=int)

    if n == 0 or spend_budget <= 0:
        return alloc

    # One vectorized call — pass spend_budget as the ceiling so the clip
    # never masks the true prediction.
    predicted = count_model.predict(frontier, np.full(n, spend_budget, dtype=int))

    # Allocate greedily: highest predicted count first, up to its prediction.
    order = np.argsort(-predicted)
    remaining = spend_budget
    for i in order:
        give = min(int(predicted[i]), remaining)
        alloc[i] = give
        remaining -= give
        if remaining <= 0:
            break

    # Spread any leftover (zero marginal gain) round-robin.
    if remaining > 0:
        for i in order:
            alloc[i] += 1
            remaining -= 1
            if remaining <= 0:
                break

    return alloc
