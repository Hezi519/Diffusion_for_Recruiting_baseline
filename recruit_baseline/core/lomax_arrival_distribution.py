# Standard library imports

# Third-party imports
from scipy.stats import lomax

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution

"""
Lomax (Pareto Type II) distribution over natural numbers
"""
class LomaxArrivalDistribution(AbstractArrivalDistribution):
    def __init__(self, rng_seed: int, c: float) -> None:
        super().__init__(rng_seed)
        self.params['c'] = c
        self.c = c

    """
    Lomax has support on [0, inf). Round down when sampling.
    """
    def sample(self) -> int:
        outcome = int(lomax.rvs(self.c, random_state=self.rng))
        return outcome

    """
    Since Lomax is continuous distribution, compute cdf(k+1) - cdf(k)
    According to ChatGPT, should use sf(k) - sf(k+1) for better numerical stability
    """
    def prob_equal(self, k: int) -> float:
        if k < 0:
            return 0.0
        else:
            # return float(lomax.cdf(k+1, self.c) - lomax.cdf(k, self.c))
            return float(lomax.sf(k, self.c) - lomax.sf(k+1, self.c))

    """
    Note: sf = 1 - cdf
    """
    def prob_at_least(self, k: int) -> float:
        if k < 0:
            return 1.0
        else:
            return float(lomax.sf(k, self.c))
