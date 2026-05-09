# Standard library imports
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution

"""
Abstract population class
Can instantiate with synthetic or real-data
"""
class AbstractPopulation(ABC):
    def __init__(self, rng_seed: int) -> None:
        self.rng = np.random.default_rng(rng_seed)

    @abstractmethod
    def sample_arrival_distributions(self, n: int) -> list[AbstractArrivalDistribution]:
        pass
