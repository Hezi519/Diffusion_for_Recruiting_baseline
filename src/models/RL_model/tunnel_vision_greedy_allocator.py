"""
TypeA-preferring allocator for the tunnel-vision experiment.

In the tunnel-vision environment, a myopic agent always allocates vouchers
to Type A nodes (high immediate rate, dead-end offspring) and ignores Type B
nodes (low immediate rate, self-replicating offspring). This captures the
"tunnel vision" failure mode of budget-DQN in the paper.

If no TypeA nodes exist in the frontier, falls back to standard greedy
allocation so the episode can continue.
"""

import numpy as np

from src.models.count_model.tunnel_vision_count_model import node_types
from src.models.RL_model.greedy_allocator import greedy_allocator


def type_a_only_allocator(state, k: int, count_model) -> np.ndarray:
    """Allocate all budget to TypeA nodes only (tunnel-vision policy).

    Distributes k budget units equally among the Type-A (LOCAL=0) frontier
    members. Any remainder is spread round-robin across the same TypeA nodes.
    TypeB and TypeC members receive zero allocation.

    If no TypeA nodes are present (e.g., all rounds after the first when
    TypeA offspring become TypeC), falls back to the standard greedy allocator
    so the episode doesn't trivially terminate.

    Args:
        state: RecruitingState with frontier_covariates.
        k: Total budget to allocate this round.
        count_model: Count model (used by fallback greedy path).

    Returns:
        (n,) integer allocation vector.
    """
    frontier = state.frontier_covariates
    n = frontier.shape[0]
    alloc = np.zeros(n, dtype=int)

    if n == 0 or k <= 0:
        return alloc

    types = node_types(frontier)
    type_a_indices = np.where(types == 0)[0]

    if len(type_a_indices) == 0:
        # No TypeA: fall back to greedy on whatever nodes remain
        return greedy_allocator(state, k, count_model)

    n_a = len(type_a_indices)
    base = k // n_a
    remainder = k % n_a

    alloc[type_a_indices] = base
    alloc[type_a_indices[:remainder]] += 1

    return alloc
