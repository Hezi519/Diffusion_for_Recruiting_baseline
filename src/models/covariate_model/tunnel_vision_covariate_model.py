"""
Type-transition covariate model for the tunnel-vision synthetic experiment.

The LOCAL covariate group encodes node type and determines the offspring type:
    Parent Type A (LOCAL=0) → child LOCAL=2 (Type C, dead-end)  with prob p_cross
    Parent Type B (LOCAL=1) → child LOCAL=1 (Type B, self-replicating) with prob p_cross
    Parent Type C (LOCAL=2/3)→ child LOCAL=2 (Type C, stays dead) with prob p_stay

All other covariate groups use per-group inheritance probabilities. By default
these are derived empirically from 73,669 ICPSR 22140 recruiter-recruit pairs
across all disease subnetworks (EMPIRICAL_INHERIT_PROBS). Pass a scalar inherit_prob to override uniformly.

This creates the "boom-bust vs sustainable" structure:
  - Allocating to A gets many immediate recruits but those recruits are dead-ends (TypeC).
  - Allocating to B gets few immediate recruits but they self-replicate (TypeB offspring).

  A myopic agent (DQN greedy) tunnel-visions on A because rate_A >> rate_B.
  A farsighted agent (Structured RL) invests in B, building a self-sustaining chain
  that produces recruits for every future round.

  Over a T-round horizon: TypeB strategy yields rate_B × T total recruits per chain,
  while TypeA strategy yields rate_A in round 1 only (then dead TypeC frontier).
  For T > rate_A / rate_B, TypeB dominates.
"""

from __future__ import annotations

import pickle

import numpy as np
from torch.utils.data import Dataset

from src.data.covariate_spec import COVARIATE_DIM, COVARIATE_GROUPS, EMPIRICAL_INHERIT_PROBS
from src.models.covariate_model.abstract_covariate_model import AbstractCovariateModel
from src.models.count_model.tunnel_vision_count_model import node_types

_LOCAL_IDX = 0
_LOCAL_NAME, _LOCAL_START, _LOCAL_END = COVARIATE_GROUPS[_LOCAL_IDX]


class TunnelVisionCovariateModel(AbstractCovariateModel):
    """Type-transition offspring model for the tunnel-vision experiment.

    Args:
        p_cross: Probability that the offspring's type follows the designed
            transition (A→C, B→B, C→C). With probability 1-p_cross the
            LOCAL group is sampled uniformly.
        inherit_prob: If None (default), uses EMPIRICAL_INHERIT_PROBS for all
            non-LOCAL groups. If a float, overrides all non-LOCAL groups uniformly.
        seed: Random seed.
    """

    def __init__(
        self,
        p_cross: float = 0.85,
        inherit_prob: float | None = None,
        seed: int = 42,
    ) -> None:
        self.p_cross = p_cross
        self.inherit_prob = inherit_prob
        self.seed = seed
        if inherit_prob is None:
            self._group_probs = {name: EMPIRICAL_INHERIT_PROBS[name] for name, _, _ in COVARIATE_GROUPS}
        else:
            self._group_probs = {name: inherit_prob for name, _, _ in COVARIATE_GROUPS}

    # A(0)→C(2), B(1)→B(1) [self-replicating], C(2)→C(2), C(3)→C(2)
    _OFFSPRING_TYPE = {0: 2, 1: 1, 2: 2, 3: 2}

    def train(self, dataset: Dataset, **kwargs) -> dict:
        return {"final_loss": 0.0}

    def sample(
        self,
        parent_covariates: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate one child per parent row with type transitions.

        LOCAL group: child type follows the designed transition with prob p_cross,
        otherwise sampled uniformly.

        All other groups: inherit parent's value with the group's empirical
        probability, otherwise sampled uniformly.

        Args:
            parent_covariates: (n, 72) parent covariate vectors.
            seed: Random seed for this call.

        Returns:
            (n, 72) valid one-hot child covariate vectors.
        """
        parent_covariates = np.asarray(parent_covariates, dtype=np.float64)
        if parent_covariates.ndim == 1:
            parent_covariates = parent_covariates[np.newaxis, :]

        rng = np.random.default_rng(seed)
        n = parent_covariates.shape[0]
        children = np.zeros((n, COVARIATE_DIM), dtype=int)

        parent_type_idx = node_types(parent_covariates)

        for g_idx, (name, start, end) in enumerate(COVARIATE_GROUPS):
            group_size = end - start

            if g_idx == _LOCAL_IDX:
                designed = np.array([self._OFFSPRING_TYPE[t] for t in parent_type_idx])
                use_cross = rng.random(n) < self.p_cross
                random_local = rng.integers(0, group_size, size=n)
                chosen = np.where(use_cross, designed, random_local)
            else:
                p = self._group_probs[name]
                parent_active = np.argmax(parent_covariates[:, start:end], axis=1)
                inherit_mask = rng.random(n) < p
                random_choice = rng.integers(0, group_size, size=n)
                chosen = np.where(inherit_mask, parent_active, random_choice)

            children[np.arange(n), start + chosen] = 1

        return children

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"p_cross": self.p_cross, "inherit_prob": self.inherit_prob, "seed": self.seed}, f)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> TunnelVisionCovariateModel:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)
