# Standard library imports

# Third-party imports

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution

"""
Uniform distribution over natural numbers from 0 to ub (inclusive)
"""
class InterpolatedArrivalDistribution(AbstractArrivalDistribution):
    def __init__(
        self,
        rng_seed: int,
        eps: float,
        original_dist: AbstractArrivalDistribution,
        noise_dist: AbstractArrivalDistribution
    ) -> None:
        super().__init__(rng_seed)
        self.params['original_dist'] = original_dist
        self.params['noise_dist'] = noise_dist
        self.eps = eps

    def sample(self) -> int:
        if self.rng.random() > self.eps:
            outcome = self.params['original_dist'].sample()
        else:
            outcome = self.params['noise_dist'].sample()
        return outcome

    def prob_equal(self, k: int) -> float:
        from_original = self.params['original_dist'].prob_equal(k)
        from_noisy = self.params['noise_dist'].prob_equal(k)
        return (1 - self.eps) * from_original + self.eps * from_noisy

    def prob_at_least(self, k: int) -> float:
        from_original = self.params['original_dist'].prob_at_least(k)
        from_noisy = self.params['noise_dist'].prob_at_least(k)
        return (1 - self.eps) * from_original + self.eps * from_noisy
