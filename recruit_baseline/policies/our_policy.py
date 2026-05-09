# Standard library imports
from dataclasses import dataclass
from functools import reduce
from typing import cast

# Third-party imports
import numpy as np

# Local imports
from core.abstract_arrival_distribution import AbstractArrivalDistribution
from core.abstract_population import AbstractPopulation
from core.population_distribution_object import PopulationDistributionObject
from core.utils import construct_polynomial_from_individual_distribution, multiply_PGFs_up_to_s, power_PGFs_up_to_s, truncate_poly
from policies.abstract_policy import AbstractPolicy

@dataclass(frozen=True)
class SurrogateObject:
    r_max: int
    gamma: float
    U: dict

def precompute_surrogate(
        r_max: int,
        gamma: float,
        population: AbstractPopulation,
        monte_carlo_samples:int = 1000
    ) -> SurrogateObject:
    q = PopulationDistributionObject(r_max+1, population, monte_carlo_samples)
    U_value = dict()

    # Base cases: U(0, n)=0 and U(r, 0)=0
    for r in range(r_max + 1):
        U_value[(r, 0)] = 0.0
    for n in range(r_max + 1):
        U_value[(0, n)] = 0.0

    # Fill bottom-up: increasing r, then n
    for r in range(1, r_max + 1):
        for n in range(1, r_max + 1):
            best_value = -np.inf
            for s in range(r + 1):
                # Even allocation
                # Everyone get a = floor(s/n) resources, then c = s - a * n individuals gets one extra resource
                a = s // n
                c = s - a * n

                # Build F(n, s)
                F_n_s = n * np.sum(q.coeffs_at_least[1:a+1])
                if c > 0:
                    F_n_s += c * q.coeffs_at_least[a+1]

                # Build H_a^(n-c) and H_a_plus_1^(c)
                # Merge degrees larger than s
                H_a = q.get_coeff(a)
                combined_poly = power_PGFs_up_to_s(H_a, n - c, s)
                if c > 0:
                    H_a_plus_1 = q.get_coeff(a+1)
                    combined_poly = multiply_PGFs_up_to_s(
                        combined_poly,
                        power_PGFs_up_to_s(H_a_plus_1, c, s),
                        s
                    )

                # Compute value for s
                value = F_n_s + gamma * sum([
                    combined_poly[m] * U_value[(r-s, m)]
                    for m in range(s+1)
                ])
                
                # Track best
                best_value = max(value, best_value)
            U_value[(r, n)] = best_value

    return SurrogateObject(r_max, gamma, U_value)

class OurPolicy(AbstractPolicy):
    def __init__(
        self,
        rng_seed: int,
        gamma: float,
        population: AbstractPopulation,
        surrogate_object: SurrogateObject
    ) -> None:
        super().__init__(rng_seed, gamma, population)
        assert np.isclose(gamma, surrogate_object.gamma)
        self.r_max = surrogate_object.r_max
        self.U = surrogate_object.U
    
    def __repr__(self):
        return "OurPolicy"

    def generate_assignments(
        self,
        available_budget: int,
        distributions_or_realizations: list[AbstractArrivalDistribution | int]
    ) -> list[int]:
        distributions = cast(list[AbstractArrivalDistribution], distributions_or_realizations)
        assert available_budget <= self.r_max
        _, optimal_s_now = self.compute_U_now(available_budget, distributions)
        assignments = self.greedy_single_stage(distributions, optimal_s_now)
        return assignments

    def greedy_single_stage(
        self,
        distributions: list[AbstractArrivalDistribution],
        budget: int
    ) -> list[int]:
        n = len(distributions)
        assignment = [0] * n
        for _ in range(budget):
            idx = np.argmax(
                [distributions[i].prob_at_least(assignment[i] + 1) for i in range(n)]
            )
            assignment[idx] += 1
        return assignment

    def compute_U_now(
        self,
        r: int,
        distributions: list[AbstractArrivalDistribution]
    ) -> tuple[float, int]:
        n = len(distributions)
        assert n >= 1

        best_value = -np.inf
        best_s = 0
        for s in range(r+1):
            # Solve single-stage optimally
            assignments = self.greedy_single_stage(distributions, s)
            
            # Compute v_s(distributions)
            v_s = sum([
                distributions[i].prob_at_least(j)
                for i in range(n)
                for j in range(1, assignments[i] + 1)
            ])

            # Build prod_{i=1}^n G_{D_i}(z; assignment[i])
            # Merge degrees larger than s
            Gs = [
                construct_polynomial_from_individual_distribution(distributions[i], assignments[i])
                for i in range(n)
            ]
            combined_poly = reduce(
                lambda acc, g: multiply_PGFs_up_to_s(acc, g, s),
                Gs,
                np.array([1.0])
            )
            
            # Compute value for s
            value = v_s + self.gamma * sum([
                combined_poly[m] * self.U[(r-s, m)]
                for m in range(s+1)
            ])

            # Track best
            if value > best_value:
                best_value = value
                best_s = s
        return best_value, best_s
