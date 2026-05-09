"""
Type-based oracle count model for the tunnel-vision synthetic experiment.

Node type is encoded in the first covariate group (LOCAL, dim 0-3):
    LOCAL=0  →  Type A  (boom-bust):    high immediate rate, offspring are TypeC (dead-ends)
    LOCAL=1  →  Type B  (sustainable):  low immediate rate,  offspring are TypeB (self-replicating)
    LOCAL=2/3→  Type C  (dead-end):     near-zero rate,      offspring stay TypeC

Rates are deterministic by type — no weight vector, no fitting required.

Design intent:
    A myopic agent (e.g. DQN greedy) concentrates vouchers on TypeA because
    rate_A >> rate_B. This produces many recruits in round 1 but then the frontier
    collapses to TypeC (dead). Over T rounds the myopic strategy earns ≈ rate_A once.

    A farsighted agent (Structured RL) invests in TypeB despite lower immediate rate,
    building a self-replicating chain that produces rate_B recruits every round.
    Over T rounds it earns ≈ rate_B × T, which dominates for T > rate_A / rate_B.
"""

from __future__ import annotations

import numpy as np

from src.data.covariate_spec import COVARIATE_GROUPS
from src.models.count_model.abstract_count_model import AbstractCountModel

# LOCAL group is the first covariate group: (name, start=0, end=4)
_LOCAL_START, _LOCAL_END = COVARIATE_GROUPS[0][1], COVARIATE_GROUPS[0][2]

# Poisson rate per type (LOCAL category index)
_TYPE_RATES = {
    0: 4.0,   # Type A: high immediate rate
    1: 0.8,   # Type B: low immediate, but offspring are Type A
    2: 0.02,  # Type C: dead end
    3: 0.02,  # Type C: dead end
}


def node_types(covariates: np.ndarray) -> np.ndarray:
    """Return per-row LOCAL category index (0-3)."""
    return np.argmax(covariates[:, _LOCAL_START:_LOCAL_END], axis=1)


class TunnelVisionCountModel(AbstractCountModel):
    """Oracle Poisson count model with type-dependent rates.

    The rate is determined solely by the LOCAL covariate group (type).
    All other covariate dimensions are ignored for the rate calculation,
    making the type signal unambiguous.

    Args:
        rate_a: Poisson rate for Type A (boom-bust) nodes.
        rate_b: Poisson rate for Type B (slow-burn) nodes.
        rate_c: Poisson rate for Type C (dead-end) nodes.
        seed: Random seed.
    """

    def __init__(
        self,
        rate_a: float = 4.0,
        rate_b: float = 0.8,
        rate_c: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.rate_a = rate_a
        self.rate_b = rate_b
        self.rate_c = rate_c
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._type_rates = {0: rate_a, 1: rate_b, 2: rate_c, 3: rate_c}

    def _rates(self, covariates: np.ndarray) -> np.ndarray:
        types = node_types(covariates)
        return np.array([self._type_rates[t] for t in types])

    def predict(
        self,
        frontier_covariates: np.ndarray,
        allocations: np.ndarray,
    ) -> np.ndarray:
        frontier_covariates = np.asarray(frontier_covariates, dtype=np.float64)
        allocations = np.asarray(allocations, dtype=int)
        rates = self._rates(frontier_covariates)
        counts = self._rng.poisson(rates).astype(int)
        return np.clip(counts, 0, allocations)
