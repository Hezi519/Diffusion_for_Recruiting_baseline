# Standard library imports

# Third-party imports
import numpy as np

# Local imports
from core.abstract_population import AbstractPopulation

class PopulationDistributionObject:
    def __init__(
        self,
        tail_threshold: int,
        population: AbstractPopulation,
        monte_carlo_samples: int = 1000
    ) -> None:
        """
        Population-level mixture distribution: D ~ P, then X ~ D
        We use Monte Carlo approximation for Pr[X = j] for j = 0, 1, ..., tail_threshold-1
        Then, the last bucket is all the remaining probability mass,
            i.e.,  Pr[X = tail_threshold] = Pr[X >= tail_threshold] = 1 - sum_j Pr[X = j] 
        """
        self.tail_threshold = tail_threshold
        distributions = population.sample_arrival_distributions(monte_carlo_samples)

        pmf_prefix = []
        for j in range(tail_threshold):
            pmf_prefix.append(np.mean([dist.prob_equal(j) for dist in distributions]))

        # In case Monte Carlo makes prefix sum > 1, clamp leftover to 0 and normalize to 1
        tail = max(0.0, 1.0 - float(np.sum(pmf_prefix)))
        self.coeffs_equal = np.array(pmf_prefix + [tail])
        self.coeffs_equal /= self.coeffs_equal.sum()

        self.coeffs_at_least = np.cumsum(self.coeffs_equal[::-1])[::-1]
            
    def get_coeff(self, k: int) -> np.ndarray:
        assert 0 <= k and k <= self.tail_threshold
        coeffs = np.zeros(k+1, dtype=float)
        coeffs[:k] = self.coeffs_equal[:k]
        coeffs[k] = self.coeffs_at_least[k]
        return coeffs


# class PopulationDistributionObject:
#     def __init__(
#         self,
#         max_degree: int,
#         population: AbstractPopulation,
#         monte_carlo_samples: int = 1000
#     ) -> None:
#         """
#         Population-level mixture distribution: D ~ P, then X ~ D
#         We use Monte Carlo approximation for Pr[X = j] for j = 0, 1, ..., max_degree-1
#         Then, the last bucket is all the remaining probability mass,
#             i.e.,  Pr[X = max_degree] = Pr[X >= max_degree] = 1 - sum_j Pr[X = j] 
#         """
#         self.max_degree = max_degree
#         distributions = population.sample_arrival_distributions(monte_carlo_samples)
#         self.coeffs_equal = [
#             sum([
#                 distributions[i].prob_equal(j)
#                 for i in range(monte_carlo_samples)
#             ]) / monte_carlo_samples
#             for j in range(max_degree)
#         ]

#         # In case Monte Carlo makes prefix sum > 1, clamp leftover to 0 and normalize to 1
#         leftover = max(0.0, 1 - sum(self.coeffs_equal))
#         self.coeffs_equal = np.array(self.coeffs_equal + [leftover])
#         self.coeffs_equal = self.coeffs_equal / np.sum(self.coeffs_equal)

#         self.coeffs_at_least = np.cumsum(self.coeffs_equal[::-1])[::-1]
            
#     def get_coeff(self, k: int) -> np.ndarray:
#         assert 0 <= k and k <= self.max_degree
#         coeffs = np.zeros(k+1, dtype=float)
#         coeffs[:k] = self.coeffs_equal[:k]
#         coeffs[k] = self.coeffs_at_least[k]
#         return coeffs
