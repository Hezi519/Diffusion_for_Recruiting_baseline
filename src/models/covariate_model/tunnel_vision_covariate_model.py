"""
Type-transition covariate model for the tunnel-vision synthetic experiment.

The LOCAL covariate group encodes node type and determines the offspring type:
    Parent Type A (LOCAL=0) → child LOCAL=2 (Type C, dead-end)  with prob p_cross
    Parent Type B (LOCAL=1) → child LOCAL=1 (Type B, self-replicating) with prob p_cross
    Parent Type C (LOCAL=2/3)→ child LOCAL=2 (Type C, stays dead) with prob p_stay

All other covariate groups are inherited normally (prob inherit_prob per group).

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

from src.data.covariate_spec import COVARIATE_DIM, COVARIATE_GROUPS
from src.models.covariate_model.abstract_covariate_model import AbstractCovariateModel
from src.models.count_model.tunnel_vision_count_model import node_types

_LOCAL_IDX = 0   # index of LOCAL group in COVARIATE_GROUPS
_LOCAL_NAME, _LOCAL_START, _LOCAL_END = COVARIATE_GROUPS[_LOCAL_IDX]


class TunnelVisionCovariateModel(AbstractCovariateModel):
    """Type-transition offspring model for the tunnel-vision experiment.

    Args:
        p_cross: Probability that the offspring's type follows the designed
            transition (A→C, B→A, C→C). With probability 1-p_cross the
            LOCAL group is sampled uniformly (exploration noise).
        inherit_prob: Per-group inheritance probability for all covariate
            groups *other* than LOCAL. Matches SyntheticCovariateModel.
        seed: Random seed.
    """

    def __init__(
        self,
        p_cross: float = 0.85,
        inherit_prob: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.p_cross = p_cross
        self.inherit_prob = inherit_prob
        self.seed = seed

    # Designed offspring type per parent type (LOCAL category index)
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

        For the LOCAL group: with prob p_cross, the child's type follows
        the designed transition. Otherwise the LOCAL group is drawn uniformly.

        For all other groups: inherit parent's value with prob inherit_prob,
        otherwise sample uniformly.

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

        for g_idx, (_, start, end) in enumerate(COVARIATE_GROUPS):
            group_size = end - start

            if g_idx == _LOCAL_IDX:
                # Designed type transition for the LOCAL group
                designed = np.array(
                    [self._OFFSPRING_TYPE[t] for t in parent_type_idx]
                )
                use_cross = rng.random(n) < self.p_cross
                random_local = rng.integers(0, group_size, size=n)
                chosen = np.where(use_cross, designed, random_local)
            else:
                # Normal inheritance for all other groups
                parent_active = np.argmax(parent_covariates[:, start:end], axis=1)
                inherit_mask = rng.random(n) < self.inherit_prob
                random_choice = rng.integers(0, group_size, size=n)
                chosen = np.where(inherit_mask, parent_active, random_choice)

            children[np.arange(n), start + chosen] = 1

        return children

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"p_cross": self.p_cross, "inherit_prob": self.inherit_prob, "seed": self.seed},
                f,
            )

    @classmethod
    def load(cls, path: str, device: str = "auto") -> TunnelVisionCovariateModel:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)
