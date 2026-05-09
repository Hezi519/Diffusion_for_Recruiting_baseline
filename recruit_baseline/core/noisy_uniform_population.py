# Standard library imports

# Third-party imports

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution
from core.abstract_population import AbstractPopulation
from core.empirical_arrival_distribution import EmpiricalArrivalDistribution

class NoisyUniformPopulation(AbstractPopulation):
    def __init__(self, rng_seed: int, ub: int, eps: float) -> None:
        super().__init__(rng_seed)
        self.ub = ub
        self.eps = eps

    def sample_arrival_distributions(self, n: int) -> list[AbstractArrivalDistribution]:
        # Add random noise of magnitude +/- eps/(ub+1) at each coordinate
        distributions = []
        for _ in range(n):
            rng = int(self.rng.integers(int(1e6)))
            signs = [2 * (self.rng.random() > 0.5) - 1 for _ in range(self.ub+1)]
            dist = {i: (1.0 + signs[i] * self.eps)/(self.ub + 1) for i in range(self.ub+1)}
            total = sum(dist.values())
            normalized_dist = {k: v/total for k,v in dist.items()}
            distributions.append(EmpiricalArrivalDistribution(rng, normalized_dist))
        return distributions
