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

    for _ in range(spend_budget):

        counts_now = count_model.predict(frontier, alloc)

        scores = np.zeros(n)

        for i in range(n):
            alloc[i] += 1
            counts_new = count_model.predict(frontier, alloc)
            scores[i] = counts_new[i] - counts_now[i]
            alloc[i] -= 1  # restore

        best_i = int(np.argmax(scores))
        alloc[best_i] += 1

    return alloc