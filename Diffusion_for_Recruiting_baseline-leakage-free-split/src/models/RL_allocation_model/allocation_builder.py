from __future__ import annotations

import numpy as np


def _safe_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        return x.copy()

    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    denom = np.sum(exp_x)

    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(x, dtype=np.float64) / float(x.size)

    return exp_x / denom


def _largest_remainder_rounding(raw_alloc: np.ndarray, budget: int) -> np.ndarray:
    raw_alloc = np.asarray(raw_alloc, dtype=np.float64)

    if budget <= 0 or raw_alloc.size == 0:
        return np.zeros_like(raw_alloc, dtype=int)

    floored = np.floor(raw_alloc).astype(int)
    remainder = int(budget - floored.sum())

    if remainder < 0:
        order = np.argsort(-floored)
        for idx in order[:(-remainder)]:
            floored[idx] -= 1
        remainder = 0

    if remainder > 0:
        frac = raw_alloc - floored
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            floored[idx] += 1

    return floored


def build_allocation(
    budget: int,
    k: int,
    scores: np.ndarray,
) -> np.ndarray:
    """
    Build integer allocation:
        1. choose top-k nodes by score
        2. softmax scores inside top-k
        3. allocate budget proportionally
        4. round while preserving exact budget
    """
    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")

    n = scores.shape[0]

    if budget < 0:
        raise ValueError(f"budget must be nonnegative, got {budget}")
    if k < 0:
        raise ValueError(f"k must be nonnegative, got {k}")

    if n == 0 or budget == 0 or k == 0:
        return np.zeros(n, dtype=int)

    budget = int(budget)
    k = min(int(k), n)

    top_idx = np.argpartition(scores, -k)[-k:]
    top_scores = scores[top_idx]

    weights = _safe_softmax(top_scores)
    raw_alloc_top = budget * weights
    alloc_top = _largest_remainder_rounding(raw_alloc_top, budget)

    alloc = np.zeros(n, dtype=int)
    alloc[top_idx] = alloc_top

    if np.any(alloc < 0):
        raise RuntimeError("Allocation builder produced negative allocation.")
    if int(alloc.sum()) != budget:
        raise RuntimeError(
            f"Allocation sum mismatch: got {alloc.sum()}, expected {budget}"
        )
    if np.count_nonzero(alloc) > k:
        raise RuntimeError(
            f"Too many active nodes: got {np.count_nonzero(alloc)}, expected <= {k}"
        )

    return alloc