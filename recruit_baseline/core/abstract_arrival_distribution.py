# Standard library imports
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np

# Local imports

"""
Discrete distribution over natural numbers
"""
class AbstractArrivalDistribution(ABC):
    def __init__(self, rng_seed: int) -> None:
        self.rng = np.random.default_rng(rng_seed)
        self.params = dict()
        self.params['rng_seed'] = rng_seed

    """
    Returns a realization according to the arrival distribution
    """
    @abstractmethod
    def sample(self) -> int:
        pass

    """
    Returns Pr(X = k) for natural number k
    """
    @abstractmethod
    def prob_equal(self, k: int) -> float:
        pass

    """
    Returns Pr(X >= k) for natural number k
    """
    @abstractmethod
    def prob_at_least(self, k: int) -> float:
        pass