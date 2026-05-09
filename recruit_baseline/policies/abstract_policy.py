# Standard library imports
from abc import ABC, ABCMeta, abstractmethod
from copy import deepcopy

# Third-party imports
import numpy as np

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution
from core.abstract_population import AbstractPopulation

class PolicyMeta(ABCMeta):
    def __repr__(cls):
        return cls.__name__
    
"""
Abstract policy class
Can instantiate with constant policy or our own policy
"""
class AbstractPolicy(ABC, metaclass=PolicyMeta):
    def __init__(
        self,
        rng_seed: int,
        gamma: float,
        population: AbstractPopulation,
    ) -> None:
        self.rng = np.random.default_rng(rng_seed)
        self.gamma = gamma
        self.population = deepcopy(population)

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def generate_assignments(
        self,
        available_budget: int,
        distributions_or_realizations: list[AbstractArrivalDistribution | int]
    ) -> list[int]:
        pass
