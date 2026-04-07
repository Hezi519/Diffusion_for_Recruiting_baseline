import numpy as np


def greedy_allocator(state, spend_budget: int, count_model) -> np.ndarray:
    """
    Greedy allocation based on marginal expected recruits.

    Args:
        state: RecruitingState
        spend_budget: total budget to allocate this round
        count_model: your count model (must have predict())

    Returns:
        alloc_vec: (n_t,) integer allocation
    """

    frontier = state.frontier_covariates   # shape (n, d)
    n = frontier.shape[0]

    alloc = np.zeros(n, dtype=int)

    if n == 0 or spend_budget <= 0:
        return alloc

    # Precompute baseline predicted counts (unclipped by allocation)
    raw_counts = count_model.predict(frontier, np.full(n, spend_budget, dtype=int))

    for _ in range(spend_budget):
        # Marginal gain: how many more recruits does person i get with +1 allocation?
        # Use raw predicted counts clipped to current vs +1 allocation
        counts_now = np.clip(raw_counts, 0, alloc)
        counts_plus1 = np.clip(raw_counts, 0, alloc + 1)
        scores = counts_plus1 - counts_now

        best_i = int(np.argmax(scores))
        alloc[best_i] += 1

    return alloc