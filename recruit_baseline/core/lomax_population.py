# Standard library imports

# Third-party imports

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution
from core.abstract_population import AbstractPopulation
from core.lomax_arrival_distribution import LomaxArrivalDistribution

"""
Synthetic population based on Lomax (Pareto Type II) distribution
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lomax.html
"""
class LomaxPopulation(AbstractPopulation):
    def __init__(self, rng_seed: int) -> None:
        super().__init__(rng_seed)

    def sample_arrival_distributions(self, n: int) -> list[AbstractArrivalDistribution]:
        # According to ChatGPT,
        # c = 1.5 ~ 3 is typical of typical for social and communication networks
        # It then suggested using Gamma(2.5, 0.5) distribution to sample the parameter c for Lomax
        distributions: list[AbstractArrivalDistribution] = [
            LomaxArrivalDistribution(
                int(self.rng.integers(int(1e6))),
                self.rng.gamma(shape=2.5, scale=0.5)
            )
            for _ in range(n)
        ]
        return distributions
