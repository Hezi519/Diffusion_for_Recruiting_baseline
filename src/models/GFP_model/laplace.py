from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonteCarloLaplaceProvider:
    """
    Cached Monte Carlo estimate of alpha(x) = E[exp(-h(child)) | x].

    The planner can then use alpha without online rollouts inside the greedy
    allocation loop. Cache keys are exact covariate rows, which is effective for
    the one-hot synthetic environments.
    """

    def __init__(
        self,
        covariate_model,
        value_surrogate,
        n_samples: int = 64,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ) -> None:
        self.covariate_model = covariate_model
        self.value_surrogate = value_surrogate
        self.n_samples = max(1, int(n_samples))
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self._cache: dict[bytes, np.ndarray] = {}

    @staticmethod
    def _key(row: np.ndarray) -> bytes:
        row = np.asarray(row, dtype=np.float64)
        return hashlib.blake2b(row.tobytes(), digest_size=16).digest()

    def clear_cache(self) -> None:
        self._cache.clear()

    def fork_for_value_surrogate(self, value_surrogate, seed: int):
        return MonteCarloLaplaceProvider(
            covariate_model=self.covariate_model,
            value_surrogate=value_surrogate,
            n_samples=self.n_samples,
            seed=seed,
            device=self.device,
        )

    @torch.no_grad()
    def alpha(self, parent_covariates: np.ndarray) -> np.ndarray:
        parent_covariates = np.asarray(parent_covariates, dtype=np.float64)
        if parent_covariates.ndim == 1:
            parent_covariates = parent_covariates[np.newaxis, :]

        out: list[np.ndarray] = []
        missing_rows = []
        missing_positions = []
        for pos, row in enumerate(parent_covariates):
            key = self._key(row)
            if key in self._cache:
                out.append(self._cache[key])
            else:
                out.append(None)  # type: ignore[arg-type]
                missing_rows.append(row)
                missing_positions.append(pos)

        if missing_rows:
            repeated = np.repeat(np.asarray(missing_rows), self.n_samples, axis=0)
            children = self.covariate_model.sample(
                repeated,
                seed=int(self.rng.integers(1 << 31)),
            )
            child_tensor = torch.tensor(
                children,
                dtype=torch.float32,
                device=self.device,
            )
            h = self.value_surrogate.coverage(child_tensor)
            exp_h = torch.exp(-h).detach().cpu().numpy()

            latent_dim = exp_h.shape[1]
            exp_h = exp_h.reshape(len(missing_rows), self.n_samples, latent_dim)
            estimates = np.clip(exp_h.mean(axis=1), 1e-8, 1.0)

            for row, pos, est in zip(missing_rows, missing_positions, estimates):
                self._cache[self._key(row)] = est
                out[pos] = est

        return np.vstack(out).astype(np.float64)


class _MomentNet(nn.Module):
    def __init__(self, covariate_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class AmortizedLaplaceProvider:
    """
    Moment network L_eta(x) ≈ E[exp(-h_phi(child)) | parent=x].

    This is the amortized alternative described in the paper. It trains from
    samples of the learned/offline offspring model, then online planning only
    queries the moment network.
    """

    def __init__(
        self,
        covariate_model,
        value_surrogate,
        parent_pool: np.ndarray,
        n_train_parents: int = 1024,
        n_child_samples: int = 16,
        train_steps: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        hidden_dim: int = 64,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ) -> None:
        self.covariate_model = covariate_model
        self.value_surrogate = value_surrogate
        self.parent_pool = np.asarray(parent_pool, dtype=np.float64)
        self.n_train_parents = max(1, int(n_train_parents))
        self.n_child_samples = max(1, int(n_child_samples))
        self.train_steps = max(1, int(train_steps))
        self.batch_size = max(1, int(batch_size))
        self.lr = float(lr)
        self.hidden_dim = int(hidden_dim)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self.moment_net = _MomentNet(
            covariate_dim=self.parent_pool.shape[1],
            latent_dim=value_surrogate.latent_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

    def clear_cache(self) -> None:
        pass

    def fork_for_value_surrogate(self, value_surrogate, seed: int):
        provider = AmortizedLaplaceProvider(
            covariate_model=self.covariate_model,
            value_surrogate=value_surrogate,
            parent_pool=self.parent_pool,
            n_train_parents=self.n_train_parents,
            n_child_samples=self.n_child_samples,
            train_steps=max(10, self.train_steps // 2),
            batch_size=self.batch_size,
            lr=self.lr,
            hidden_dim=self.hidden_dim,
            seed=seed,
            device=self.device,
        )
        provider.refresh()
        return provider

    @torch.no_grad()
    def _targets(self, parents: np.ndarray) -> np.ndarray:
        repeated = np.repeat(parents, self.n_child_samples, axis=0)
        children = self.covariate_model.sample(
            repeated,
            seed=int(self.rng.integers(1 << 31)),
        )
        child_tensor = torch.tensor(children, dtype=torch.float32, device=self.device)
        h = self.value_surrogate.coverage(child_tensor)
        exp_h = torch.exp(-h).detach().cpu().numpy()
        exp_h = exp_h.reshape(parents.shape[0], self.n_child_samples, -1)
        return np.clip(exp_h.mean(axis=1), 1e-8, 1.0)

    def refresh(self) -> dict:
        idx = self.rng.integers(
            0,
            self.parent_pool.shape[0],
            size=self.n_train_parents,
        )
        parents = self.parent_pool[idx]
        targets = self._targets(parents)

        x = torch.tensor(parents, dtype=torch.float32, device=self.device)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.moment_net.parameters(), lr=self.lr)
        losses: list[float] = []

        for _ in range(self.train_steps):
            batch_idx = torch.randint(
                low=0,
                high=x.shape[0],
                size=(min(self.batch_size, x.shape[0]),),
                device=self.device,
            )
            pred = self.moment_net(x[batch_idx])
            loss = F.mse_loss(pred, y[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        return {
            "final_loss": losses[-1] if losses else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "n_train_parents": int(self.n_train_parents),
        }

    @torch.no_grad()
    def alpha(self, parent_covariates: np.ndarray) -> np.ndarray:
        parent_covariates = np.asarray(parent_covariates, dtype=np.float64)
        if parent_covariates.ndim == 1:
            parent_covariates = parent_covariates[np.newaxis, :]
        x = torch.tensor(parent_covariates, dtype=torch.float32, device=self.device)
        alpha = self.moment_net(x).detach().cpu().numpy()
        return np.clip(alpha, 1e-8, 1.0).astype(np.float64)
