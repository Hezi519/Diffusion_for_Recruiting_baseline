"""Adaptive surrogate policy for the diffusion recruiting environment.

This module adapts the surrogate dynamic program from ``recruit_baseline`` to
the diffusion recruiting environment, where the online state is a frontier of
covariate vectors instead of explicit arrival-distribution objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import erf, sqrt

import numpy as np


def _normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 1e-12:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + erf((x - mean) / (std * sqrt(2.0))))


def _rounded_gaussian_capacity_pmf(
    mean: float,
    std: float,
    max_support: int,
) -> np.ndarray:
    """Discretize a Gaussian latent count into nonnegative integer capacity.

    Bucket ``max_support`` stores all tail mass ``Pr[count >= max_support]``.
    The environment later clips realised recruits by the allocation, matching
    the original model's ``min(D_i, a_i)`` interpretation.
    """
    if max_support < 0:
        raise ValueError("max_support must be nonnegative")

    if std <= 1e-12:
        value = int(round(mean))
        value = max(0, min(max_support, value))
        pmf = np.zeros(max_support + 1, dtype=float)
        pmf[value] = 1.0
        return pmf

    pmf = np.zeros(max_support + 1, dtype=float)
    pmf[0] = _normal_cdf(0.5, mean, std)
    for k in range(1, max_support):
        lo = k - 0.5
        hi = k + 0.5
        pmf[k] = max(0.0, _normal_cdf(hi, mean, std) - _normal_cdf(lo, mean, std))
    if max_support > 0:
        pmf[max_support] = max(0.0, 1.0 - _normal_cdf(max_support - 0.5, mean, std))

    total = float(pmf.sum())
    if total <= 0.0 or not np.isfinite(total):
        value = int(round(mean))
        value = max(0, min(max_support, value))
        pmf[:] = 0.0
        pmf[value] = 1.0
    else:
        pmf /= total
    return pmf


@dataclass(frozen=True)
class DiscreteArrivalDistribution:
    """Discrete count-capacity distribution with a final tail bucket."""

    pmf: np.ndarray

    def __post_init__(self) -> None:
        pmf = np.asarray(self.pmf, dtype=float)
        if pmf.ndim != 1 or len(pmf) == 0:
            raise ValueError("pmf must be a nonempty 1-D array")
        total = float(pmf.sum())
        if total <= 0.0:
            raise ValueError("pmf must have positive mass")
        object.__setattr__(self, "pmf", pmf / total)
        object.__setattr__(self, "_tail", np.cumsum((pmf / total)[::-1])[::-1])

    @property
    def max_support(self) -> int:
        return len(self.pmf) - 1

    def prob_equal(self, k: int) -> float:
        if k < 0:
            return 0.0
        if k > self.max_support:
            return 0.0
        return float(self.pmf[k])

    def prob_at_least(self, k: int) -> float:
        if k <= 0:
            return 1.0
        if k > self.max_support:
            return 0.0
        return float(self._tail[k])


class GaussianCountDistributionAdapter:
    """Turns a fitted GaussianCountModel into discrete capacity distributions."""

    def __init__(
        self,
        count_model,
        max_support: int,
        min_std: float = 1e-3,
        std_scale: float = 1.0,
    ) -> None:
        if max_support < 0:
            raise ValueError("max_support must be nonnegative")
        self.count_model = count_model
        self.max_support = int(max_support)
        self.min_std = float(min_std)
        self.std_scale = float(std_scale)

    def distributions_for_covariates(self, covariates: np.ndarray) -> list[DiscreteArrivalDistribution]:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim != 2:
            raise ValueError(f"covariates must be 2-D, got {covariates.shape}")
        means, stds = self.count_model.gpr.predict(covariates, return_std=True)
        stds = np.maximum(np.asarray(stds, dtype=float) * self.std_scale, self.min_std)
        return [
            DiscreteArrivalDistribution(
                _rounded_gaussian_capacity_pmf(float(mu), float(sig), self.max_support)
            )
            for mu, sig in zip(means, stds)
        ]

    def population_pmf(self, covariates: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim != 2:
            raise ValueError(f"covariates must be 2-D, got {covariates.shape}")
        if covariates.shape[0] == 0:
            raise ValueError("cannot build a population distribution from zero covariates")

        rng = np.random.default_rng(seed)
        n = min(int(sample_size), covariates.shape[0])
        idx = rng.choice(covariates.shape[0], size=n, replace=False)
        dists = self.distributions_for_covariates(covariates[idx])
        pmf = np.mean([dist.pmf for dist in dists], axis=0)
        pmf = np.asarray(pmf, dtype=float)
        pmf /= pmf.sum()
        return pmf


@dataclass(frozen=True)
class SurrogateObject:
    r_max: int
    gamma: float
    U: dict[tuple[int, int], float]
    population_pmf: np.ndarray


def truncate_poly(poly: np.ndarray, s: int) -> np.ndarray:
    out = np.zeros(s + 1, dtype=float)
    k = min(s + 1, len(poly))
    out[:k] = poly[:k]
    if len(poly) > s + 1:
        out[s] += float(poly[s + 1 :].sum())
    total = float(out.sum())
    if total > 0.0:
        out /= total
    return out


def multiply_pgfs_up_to_s(pgf1: np.ndarray, pgf2: np.ndarray, s: int) -> np.ndarray:
    return truncate_poly(np.polynomial.polynomial.polymul(pgf1, pgf2), s)


def power_pgf_up_to_s(pgf: np.ndarray, exponent: int, s: int) -> np.ndarray:
    result = np.array([1.0], dtype=float)
    x = truncate_poly(pgf, s)
    e = int(exponent)
    while e > 0:
        if e & 1:
            result = multiply_pgfs_up_to_s(result, x, s)
        e >>= 1
        if e:
            x = multiply_pgfs_up_to_s(x, x, s)
    return truncate_poly(result, s)


def construct_polynomial_from_distribution(
    distribution: DiscreteArrivalDistribution,
    max_degree: int,
) -> np.ndarray:
    coeffs = [distribution.prob_equal(j) for j in range(max_degree)]
    coeffs.append(distribution.prob_at_least(max_degree))
    return np.asarray(coeffs, dtype=float)


def _coeff_at_least(pmf: np.ndarray) -> np.ndarray:
    return np.cumsum(pmf[::-1])[::-1]


def _coeff_for_allocation(population_pmf: np.ndarray, k: int) -> np.ndarray:
    at_least = _coeff_at_least(population_pmf)
    coeffs = np.zeros(k + 1, dtype=float)
    if k > 0:
        coeffs[:k] = population_pmf[:k]
    coeffs[k] = at_least[k] if k < len(at_least) else 0.0
    total = float(coeffs.sum())
    if total > 0.0:
        coeffs /= total
    return coeffs


def precompute_surrogate_from_population_pmf(
    r_max: int,
    gamma: float,
    population_pmf: np.ndarray,
) -> SurrogateObject:
    population_pmf = np.asarray(population_pmf, dtype=float)
    if len(population_pmf) < r_max + 2:
        padded = np.zeros(r_max + 2, dtype=float)
        padded[: len(population_pmf)] = population_pmf
        padded[-1] += max(0.0, 1.0 - float(padded.sum()))
        population_pmf = padded
    population_pmf = population_pmf / population_pmf.sum()
    coeffs_at_least = _coeff_at_least(population_pmf)

    U_value: dict[tuple[int, int], float] = {}
    for r in range(r_max + 1):
        U_value[(r, 0)] = 0.0
    for n in range(r_max + 1):
        U_value[(0, n)] = 0.0

    for r in range(1, r_max + 1):
        for n in range(1, r_max + 1):
            best_value = -np.inf
            for s in range(r + 1):
                a = s // n
                c = s - a * n

                F_n_s = n * float(np.sum(coeffs_at_least[1 : a + 1]))
                if c > 0:
                    F_n_s += c * float(coeffs_at_least[a + 1])

                H_a = _coeff_for_allocation(population_pmf, a)
                combined_poly = power_pgf_up_to_s(H_a, n - c, s)
                if c > 0:
                    H_a_plus_1 = _coeff_for_allocation(population_pmf, a + 1)
                    combined_poly = multiply_pgfs_up_to_s(
                        combined_poly,
                        power_pgf_up_to_s(H_a_plus_1, c, s),
                        s,
                    )

                value = F_n_s + gamma * sum(
                    combined_poly[m] * U_value[(r - s, m)] for m in range(s + 1)
                )
                best_value = max(best_value, value)
            U_value[(r, n)] = float(best_value)

    return SurrogateObject(
        r_max=int(r_max),
        gamma=float(gamma),
        U=U_value,
        population_pmf=population_pmf,
    )


class AdaptiveSurrogatePolicy:
    """Online adaptive surrogate allocator for RecruitingState objects."""

    def __init__(
        self,
        distribution_adapter: GaussianCountDistributionAdapter,
        surrogate: SurrogateObject,
    ) -> None:
        if distribution_adapter.max_support < surrogate.r_max:
            raise ValueError("distribution support should cover the surrogate budget")
        self.distribution_adapter = distribution_adapter
        self.surrogate = surrogate
        self.gamma = surrogate.gamma

    def act(self, state) -> np.ndarray:
        if state.frontier_size == 0 or state.budget_remaining <= 0:
            return np.zeros(state.frontier_size, dtype=int)
        if state.budget_remaining > self.surrogate.r_max:
            raise ValueError(
                f"state budget {state.budget_remaining} exceeds surrogate r_max {self.surrogate.r_max}"
            )
        distributions = self.distribution_adapter.distributions_for_covariates(
            state.frontier_covariates
        )
        _, optimal_s = self.compute_u_now(state.budget_remaining, distributions)
        return np.asarray(self.greedy_single_stage(distributions, optimal_s), dtype=int)

    def greedy_single_stage(
        self,
        distributions: list[DiscreteArrivalDistribution],
        budget: int,
    ) -> list[int]:
        n = len(distributions)
        assignment = [0] * n
        for _ in range(int(budget)):
            idx = int(np.argmax([
                distributions[i].prob_at_least(assignment[i] + 1)
                for i in range(n)
            ]))
            assignment[idx] += 1
        return assignment

    def compute_u_now(
        self,
        r: int,
        distributions: list[DiscreteArrivalDistribution],
    ) -> tuple[float, int]:
        n = len(distributions)
        if n == 0:
            return 0.0, 0

        best_value = -np.inf
        best_s = 0
        n_cap = self.surrogate.r_max
        for s in range(int(r) + 1):
            assignments = self.greedy_single_stage(distributions, s)
            v_s = sum(
                distributions[i].prob_at_least(j)
                for i in range(n)
                for j in range(1, assignments[i] + 1)
            )
            Gs = [
                construct_polynomial_from_distribution(distributions[i], assignments[i])
                for i in range(n)
            ]
            combined_poly = reduce(
                lambda acc, g: multiply_pgfs_up_to_s(acc, g, s),
                Gs,
                np.array([1.0], dtype=float),
            )
            value = v_s + self.gamma * sum(
                combined_poly[m] * self.surrogate.U[(r - s, min(m, n_cap))]
                for m in range(s + 1)
            )
            if value > best_value:
                best_value = float(value)
                best_s = int(s)
        return best_value, best_s
