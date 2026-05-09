# Standard library imports

# Third-party imports
import numpy as np

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution

"""
Empirical neighbor distribution from some real-world dataset over natural numbers
"""
class EmpiricalArrivalDistribution(AbstractArrivalDistribution):
    def __init__(self, rng_seed: int, empirical_distribution: dict) -> None:
        super().__init__(rng_seed)
        self.params['pdf'] = empirical_distribution
        self.pdf = np.zeros(max(empirical_distribution.keys()) + 1)
        for degree, prob in empirical_distribution.items():
            self.pdf[degree] = prob
        self.cdf = np.cumsum(self.pdf)
        assert np.isclose(1, self.cdf[-1])

    def sample(self) -> int:
        return int(np.searchsorted(self.cdf, self.rng.random()))

    def prob_equal(self, k: int) -> float:
        return self.pdf[k] if k < len(self.pdf) else 0.0

    def prob_at_least(self, k: int) -> float:
        if k <= 0:
            return 1.0
        if k >= len(self.cdf):
            return 0.0
        return float(1.0 - self.cdf[k - 1])
