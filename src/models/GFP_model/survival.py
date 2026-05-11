from __future__ import annotations

import numpy as np


class PoissonCountSurvival:
    """
    Adapter exposing the GFP survival query P(C >= ell | x).

    The existing synthetic count models already keep a covariate-dependent
    Poisson rate in a private `_rates(covariates)` helper. GFP needs the
    distributional query, while the environment continues to use predict().
    """

    def __init__(self, count_model) -> None:
        if not hasattr(count_model, "_rates"):
            raise TypeError(
                "PoissonCountSurvival requires a count model with a _rates(covariates) method."
            )
        self.count_model = count_model

    def rates(self, covariates: np.ndarray) -> np.ndarray:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates[np.newaxis, :]
        rates = np.asarray(self.count_model._rates(covariates), dtype=np.float64)
        return np.maximum(rates, 0.0)

    def survival_prob(
        self,
        covariates: np.ndarray,
        ell: int | np.ndarray,
    ) -> np.ndarray:
        """
        Return P(C >= ell | x) for each row of covariates.

        Args:
            covariates: (n, d) covariate rows.
            ell: scalar or shape-(n,) positive resource index.
        """
        rates = self.rates(covariates)
        ell_arr = np.asarray(ell, dtype=int)
        if ell_arr.ndim == 0:
            ell_arr = np.full(rates.shape, int(ell_arr), dtype=int)
        else:
            ell_arr = np.broadcast_to(ell_arr, rates.shape).astype(int)

        out = np.ones_like(rates, dtype=np.float64)
        positive = ell_arr > 0
        if not np.any(positive):
            return out

        for idx in np.where(positive)[0]:
            k = int(ell_arr[idx])
            lam = float(rates[idx])
            pmf = np.exp(-lam)
            cdf = pmf
            for c in range(1, k):
                pmf *= lam / float(c)
                cdf += pmf
            out[idx] = float(np.clip(1.0 - cdf, 0.0, 1.0))
        return out

    def tau(
        self,
        covariates: np.ndarray,
        alpha: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        """
        Compute tau_ij(k_i) = E[alpha_ij ** min(k_i, C_i)].

        Args:
            covariates: (n, cov_dim)
            alpha: (n, latent_dim), values in (0, 1]
            k: (n,) integer allocations
        """
        covariates = np.asarray(covariates, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        k = np.asarray(k, dtype=int)

        n, latent_dim = alpha.shape
        tau = np.ones((n, latent_dim), dtype=np.float64)
        rates = self.rates(covariates)

        for i in range(n):
            ki = int(k[i])
            if ki <= 0:
                continue

            lam = float(rates[i])
            pmf = np.exp(-lam)
            vals = np.full(latent_dim, pmf, dtype=np.float64)
            cdf = pmf
            alpha_power = np.ones(latent_dim, dtype=np.float64)

            for c in range(1, ki):
                pmf *= lam / float(c)
                cdf += pmf
                alpha_power *= alpha[i]
                vals += pmf * alpha_power

            survival_at_k = max(0.0, 1.0 - cdf)
            vals += survival_at_k * (alpha[i] ** ki)
            tau[i] = np.clip(vals, 1e-12, 1.0)

        return tau
