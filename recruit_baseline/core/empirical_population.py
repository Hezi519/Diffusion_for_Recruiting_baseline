# Standard library imports

# Third-party imports
import numpy as np

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution
from core.abstract_population import AbstractPopulation
from core.empirical_arrival_distribution import EmpiricalArrivalDistribution

"""
Empirical neighbor population
"""
class EmpiricalPopulation(AbstractPopulation):
    def __init__(self, rng_seed: int, type_proportion: dict, empirical_distributions: dict) -> None:
        assert frozenset(type_proportion.keys()) == frozenset(empirical_distributions.keys())
        assert np.isclose(1, sum(type_proportion.values()))
        super().__init__(rng_seed)
        self.type_list = sorted(type_proportion.keys())
        self.type_p = [type_proportion[self.type_list[idx]] for idx in range(len(self.type_list))]
        self.empirical_distributions = empirical_distributions

    def sample_arrival_distributions(self, n: int) -> list[AbstractArrivalDistribution]:
        sampled_types = [int(x) for x in self.rng.choice(self.type_list, size=n, p=self.type_p)]
        distributions: list[AbstractArrivalDistribution] = [
            EmpiricalArrivalDistribution(
                int(self.rng.integers(int(1e6))),
                self.empirical_distributions[sampled_types[idx]]
            )
            for idx in range(n)
        ]
        return distributions
