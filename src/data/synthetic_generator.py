"""
Synthetic covariate pool for workshop experiments.

Provides the same interface as ICPSRGraphData for initial frontier
sampling, without requiring any external dataset.  Covariates are
sampled uniformly at random from the 72-dim one-hot schema defined
in covariate_spec.py.

Usage in a synthetic driver:

    graph_data = SyntheticGraphData(n_nodes=300, seed=42)
    initial_frontier = graph_data.sample_initial_frontier(n=5, seed=0)
    # → (5, 72) numpy array of valid one-hot covariate vectors
"""

from __future__ import annotations

import numpy as np

from src.data.covariate_spec import COVARIATE_DIM, COVARIATE_GROUPS


class SyntheticGraphData:
    """Random covariate pool for synthetic RDS experiments.

    Generates `n_nodes` random one-hot covariate vectors following
    the same 72-dim categorical schema as the ICPSR dataset.
    All nodes are treated as root seeds (no graph structure).

    Args:
        n_nodes: Number of covariate vectors to pre-generate.
            Larger pools reduce repetition across episodes.
        seed: Random seed for covariate generation.
    """

    def __init__(
        self,
        n_nodes: int = 300,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._covariates: dict[int, np.ndarray] = {}
        for i in range(n_nodes):
            self._covariates[i] = _sample_one_hot(rng)
        self._n_nodes = n_nodes

    # ------------------------------------------------------------------
    # ICPSRGraphData-compatible interface
    # ------------------------------------------------------------------

    @property
    def covariates(self) -> dict[int, np.ndarray]:
        """Node ID -> 72-dim one-hot covariate vector."""
        return self._covariates

    def sample_initial_frontier(
        self,
        n: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Sample n covariate vectors for an episode's initial frontier.

        Samples with replacement from the pre-generated pool.

        Args:
            n: Number of frontier individuals.
            seed: Random seed for this call.

        Returns:
            (n, 72) array of valid one-hot covariate vectors.
        """
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, self._n_nodes, size=n)
        return np.array([self._covariates[i] for i in indices], dtype=np.float64)


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _sample_one_hot(rng: np.random.Generator) -> np.ndarray:
    """Draw a single random 72-dim one-hot vector."""
    vec = np.zeros(COVARIATE_DIM, dtype=int)
    for _, start, end in COVARIATE_GROUPS:
        vec[rng.integers(start, end)] = 1
    return vec
