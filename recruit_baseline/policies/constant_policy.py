# Standard library imports

# Third-party imports

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution
from core.abstract_population import AbstractPopulation
from policies.abstract_policy import AbstractPolicy

class ConstantPolicy(AbstractPolicy):
    def __init__(
        self,
        rng_seed: int,
        gamma: float,
        population: AbstractPopulation,
        const_val: int
    ) -> None:
        super().__init__(rng_seed, gamma, population)
        self.const_val = const_val
    
    def __repr__(self):
        return "ConstantPolicy"

    def generate_assignments(
        self,
        available_budget: int,
        distributions_or_realizations: list[AbstractArrivalDistribution | int]
    ) -> list[int]:
        assignments = []
        for _ in range(len(distributions_or_realizations)):
            assignments.append(max(0, min(available_budget, self.const_val)))
            available_budget -= self.const_val
        return assignments
