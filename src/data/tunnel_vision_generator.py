"""
Initial frontier generator for the tunnel-vision synthetic experiment.

Generates node pools with a controlled mix of Types A, B, C.
The type is encoded in the LOCAL covariate group (dim 0-3).

Type distribution in the pool is configurable so experiments can start with
a balanced mix of boom-bust (A) and slow-burn (B) nodes.
"""

from __future__ import annotations

import numpy as np

from src.data.covariate_spec import COVARIATE_DIM, COVARIATE_GROUPS

_LOCAL_IDX = 0
_LOCAL_NAME, _LOCAL_START, _LOCAL_END = COVARIATE_GROUPS[_LOCAL_IDX]

# LOCAL category → type name (for reference)
TYPE_LABELS = {0: "A (boom-bust)", 1: "B (sustainable)", 2: "C (dead-end)", 3: "C (dead-end)"}


class TunnelVisionGraphData:
    """Covariate pool with controlled Type A / B / C distribution.

    Args:
        n_nodes: Total number of nodes in the pool.
        type_fractions: (frac_A, frac_B, frac_C) fractions summing to 1.
            Defaults to equal A/B split with no C seeds.
        seed: Random seed.
    """

    def __init__(
        self,
        n_nodes: int = 300,
        type_fractions: tuple[float, float, float] = (0.45, 0.45, 0.10),
        seed: int = 42,
    ) -> None:
        frac_a, frac_b, frac_c = type_fractions
        assert abs(frac_a + frac_b + frac_c - 1.0) < 1e-6

        rng = np.random.default_rng(seed)
        n_a = int(n_nodes * frac_a)
        n_b = int(n_nodes * frac_b)
        n_c = n_nodes - n_a - n_b

        self._covariates: dict[int, np.ndarray] = {}
        idx = 0
        for count, local_cat in [(n_a, 0), (n_b, 1), (n_c, 2)]:
            for _ in range(count):
                self._covariates[idx] = _make_typed_cov(local_cat, rng)
                idx += 1

        self._n_a = n_a
        self._n_b = n_b
        self._n_c = n_c
        self._n_nodes = n_nodes

    @property
    def covariates(self) -> dict[int, np.ndarray]:
        return self._covariates

    def sample_initial_frontier(
        self,
        n: int,
        seed: int = 42,
        balanced: bool = True,
    ) -> np.ndarray:
        """Sample n covariate vectors for an initial frontier.

        Args:
            n: Number of frontier members.
            seed: Random seed.
            balanced: If True, ensures the frontier has equal numbers of
                Type A and Type B nodes (best for demonstrating tunnel vision).
                Any remainder is filled randomly from the pool.

        Returns:
            (n, 72) array of one-hot covariate vectors.
        """
        rng = np.random.default_rng(seed)

        if balanced:
            n_each = n // 2
            a_nodes = [i for i, c in self._covariates.items() if c[_LOCAL_START] == 1]
            b_nodes = [i for i, c in self._covariates.items() if c[_LOCAL_START + 1] == 1]
            chosen_a = rng.choice(a_nodes, size=n_each, replace=len(a_nodes) < n_each)
            chosen_b = rng.choice(b_nodes, size=n - n_each, replace=len(b_nodes) < n - n_each)
            chosen = np.concatenate([chosen_a, chosen_b])
            rng.shuffle(chosen)
        else:
            chosen = rng.integers(0, self._n_nodes, size=n)

        return np.array([self._covariates[i] for i in chosen], dtype=np.float64)

    def type_summary(self) -> str:
        return f"Pool: {self._n_a} Type-A, {self._n_b} Type-B, {self._n_c} Type-C (total {self._n_nodes})"


def _make_typed_cov(local_cat: int, rng: np.random.Generator) -> np.ndarray:
    """Make a one-hot covariate vector with a fixed LOCAL category."""
    vec = np.zeros(COVARIATE_DIM, dtype=int)
    for g_idx, (_, start, end) in enumerate(COVARIATE_GROUPS):
        if g_idx == _LOCAL_IDX:
            vec[start + local_cat] = 1
        else:
            vec[rng.integers(start, end)] = 1
    return vec
