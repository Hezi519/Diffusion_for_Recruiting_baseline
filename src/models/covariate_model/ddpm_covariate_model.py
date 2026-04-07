"""
DDPM-based covariate generation model.

This is a faithful adaptation of reference/diffusion/diffusion_model.py,
implementing the AbstractCovariateModel interface. The architecture,
hyperparameters, and DDPM math are identical to the reference.

The model learns p(child_cov | parent_cov) using a conditional DDPM:
- Forward process: noise the child covariates
- Reverse process: denoise conditioned on parent covariates
- Post-processing: argmax per covariate group to produce valid one-hot vectors
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.covariate_spec import COVARIATE_DIM, continuous_to_one_hot
from src.models.covariate_model.abstract_covariate_model import AbstractCovariateModel


# ---------------------------------------------------------------------------
# Network components (identical to reference diffusion_model.py)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sin/cos embedding for diffusion timesteps.

    Reference: diffusion_model.py lines 16-56.

    Args:
        dim: Output embedding dimension (must be even, >= 4).
        max_period: Controls minimum frequency.
        scale_by_2pi: If True, multiplies argument by 2*pi.
    """

    def __init__(self, dim: int = 16, max_period: float = 10000.0, scale_by_2pi: bool = True):
        super().__init__()
        assert dim % 2 == 0 and dim >= 4
        self.dim = dim
        self.max_period = float(max_period)
        self.scale_by_2pi = scale_by_2pi

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        t = t.to(dtype=torch.float32)

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / (half - 1)
        )
        arg = t.unsqueeze(-1) * freqs.unsqueeze(0)
        if self.scale_by_2pi:
            arg = 2.0 * math.pi * arg
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class SimpleNN(nn.Module):
    """Noise prediction network for the DDPM.

    Reference: diffusion_model.py lines 58-75.

    Input: [parent_cov(72) || noised_child_cov(72) || time_emb(16)] = 160
    Output: predicted noise (72)
    """

    def __init__(self, cov_dim: int = COVARIATE_DIM, hidden_dim: int = 512):
        super().__init__()
        input_dim = cov_dim * 2  # parent + noised child
        self.time_embedding = SinusoidalTimeEmbedding(dim=16)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cov_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 144) concatenated [parent_cov || noised_child_cov].
            t: (B, 1) normalized timestep in [0, 1].
        Returns:
            (B, 72) predicted noise.
        """
        t_emb = self.time_embedding(t)
        return self.net(torch.cat([x, t_emb], dim=-1))


# ---------------------------------------------------------------------------
# DDPM schedule and sampling (identical to reference diffusion_model.py)
# ---------------------------------------------------------------------------

@dataclass
class DDPMSchedule:
    """Linear beta schedule with derived alpha quantities.

    Reference: diffusion_model.py lines 141-146.
    """

    num_steps: int = 100
    beta_min: float = 1e-4
    beta_max: float = 0.02

    def compute(self, device: torch.device, dtype: torch.dtype = torch.float32):
        """Compute schedule tensors on the given device."""
        betas = torch.linspace(self.beta_min, self.beta_max, self.num_steps, device=device, dtype=dtype)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars


def q_sample(
    x_0: torch.Tensor,
    alpha_bars: torch.Tensor,
    t: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion process: add noise at timestep t.

    Reference: diffusion_model.py lines 148-155.

    Args:
        x_0: (B, D) clean data.
        alpha_bars: (T,) cumulative product of alphas.
        t: (B, 1) or (B,) integer timesteps.
        generator: Optional torch Generator for reproducibility.

    Returns:
        (x_t, epsilon) where x_t is noised data and epsilon is the noise added.
    """
    if t.ndim == 2 and t.size(-1) == 1:
        t = t.squeeze(-1)
    eps = torch.randn(x_0.shape, device=x_0.device, dtype=x_0.dtype, generator=generator)
    alpha_bar_t = alpha_bars[t].unsqueeze(-1)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps
    return x_t, eps


def p_sample_loop(
    model: nn.Module,
    conditions: torch.Tensor,
    schedule: DDPMSchedule,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full reverse DDPM sampling loop.

    Reference: diffusion_model.py lines 168-199.

    Args:
        model: Noise prediction network.
        conditions: (B, 72) parent covariates to condition on.
        schedule: DDPM schedule parameters.
        betas: (T,) beta values.
        alphas: (T,) alpha values.
        alpha_bars: (T,) cumulative alpha products.
        generator: Optional torch Generator.

    Returns:
        (B, 72) denoised child covariates (continuous, before one-hot).
    """
    device = conditions.device
    dtype = conditions.dtype
    n = conditions.shape[0]

    x_t = torch.randn(n, COVARIATE_DIM, device=device, dtype=dtype, generator=generator)

    for t in range(schedule.num_steps - 1, 0, -1):
        ts = torch.full((n, 1), t / (schedule.num_steps - 1), device=device, dtype=dtype)
        eps_hat = model(torch.cat([conditions, x_t], dim=1), ts)

        # DDPM posterior mean (epsilon parameterization)
        mu = (1.0 / torch.sqrt(alphas[t])) * (
            x_t - (1 - alphas[t]) / torch.sqrt(1 - alpha_bars[t]) * eps_hat
        )

        # Posterior variance
        tilde_beta_t = (1 - alpha_bars[t - 1]) / (1 - alpha_bars[t]) * betas[t]

        if t > 1:
            z = torch.randn(x_t.shape, device=device, dtype=dtype, generator=generator)
            x_t = mu + torch.sqrt(tilde_beta_t) * z
        else:
            x_t = mu  # Final step is deterministic

    return x_t


# ---------------------------------------------------------------------------
# DDPMCovariateModel — AbstractCovariateModel implementation
# ---------------------------------------------------------------------------

def _get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


class DDPMCovariateModel(AbstractCovariateModel):
    """DDPM-based covariate generation model.

    Identical architecture and hyperparameters to the reference
    diffusion_model.py. Learns p(child_cov | parent_cov) using
    a conditional denoising diffusion probabilistic model.
    """

    def __init__(
        self,
        cov_dim: int = COVARIATE_DIM,
        hidden_dim: int = 512,
        num_steps: int = 100,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        device: str = "auto",
    ) -> None:
        self.cov_dim = cov_dim
        self.hidden_dim = hidden_dim
        self.schedule = DDPMSchedule(num_steps, beta_min, beta_max)
        self.device = _get_device(device)
        self.network = SimpleNN(cov_dim, hidden_dim).to(self.device)
        # Cache schedule tensors to avoid recomputing on every sample() call
        self._betas, self._alphas, self._alpha_bars = self.schedule.compute(self.device)

    def train(
        self,
        dataset: Dataset,
        epochs: int = 4000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 42,
        log_interval: int = 100,
    ) -> dict:
        """Train the DDPM. Reference: diffusion_model.py lines 77-139."""
        loader_gen = torch.Generator(device="cpu")
        loader_gen.manual_seed(seed)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=loader_gen)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        noise_gen = torch.Generator(device=self.device)
        noise_gen.manual_seed(seed)

        betas, alphas, alpha_bars = self.schedule.compute(self.device)

        self.network.train()
        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            count = 0
            for x in data_loader:
                x = x.view(x.size(0), -1).to(self.device)

                # Sample random timestep
                t = torch.randint(
                    0, self.schedule.num_steps,
                    size=[x.size(0), 1],
                    device=self.device,
                    generator=noise_gen,
                )

                # Noise the child covariates (second half of x)
                x_child = x[:, self.cov_dim:]
                noised_child, eps = q_sample(x_child, alpha_bars, t, generator=noise_gen)

                # Normalize timestep to [0, 1]
                t_norm = t / (self.schedule.num_steps - 1.0)

                # Condition on parent covariates (first half)
                model_input = torch.cat([x[:, :self.cov_dim], noised_child], dim=1)

                # Predict noise
                predicted_eps = self.network(model_input, t_norm)

                loss = loss_fn(predicted_eps, eps)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                count += 1

            final_loss = epoch_loss / max(count, 1)
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}, Loss={final_loss:.5f}")

        print("Finished training!")
        return {"final_loss": final_loss}

    @torch.no_grad()
    def sample(
        self,
        parent_covariates: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate child covariates. Reference: diffusion_model.py lines 157-199."""
        parent_covariates = np.asarray(parent_covariates, dtype=np.float32)
        if parent_covariates.ndim == 1:
            parent_covariates = parent_covariates[np.newaxis, :]

        conditions = torch.tensor(parent_covariates, dtype=torch.float32, device=self.device)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        self.network.eval()
        x_0 = p_sample_loop(
            self.network, conditions, self.schedule,
            self._betas, self._alphas, self._alpha_bars, generator=generator,
        )

        return continuous_to_one_hot(x_0.cpu().numpy())

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "cov_dim": self.cov_dim,
            "hidden_dim": self.hidden_dim,
            "schedule": {
                "num_steps": self.schedule.num_steps,
                "beta_min": self.schedule.beta_min,
                "beta_max": self.schedule.beta_max,
            },
        }
        torch.save(checkpoint, path)
        print(f"Saved model: {path}")

    @classmethod
    def load(cls, path: str, device: str = "auto") -> DDPMCovariateModel:
        """Load model from checkpoint."""
        dev = _get_device(device)
        checkpoint = torch.load(path, map_location=dev, weights_only=True)
        model = cls(
            cov_dim=checkpoint["cov_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_steps=checkpoint["schedule"]["num_steps"],
            beta_min=checkpoint["schedule"]["beta_min"],
            beta_max=checkpoint["schedule"]["beta_max"],
            device=device,
        )
        model.network.load_state_dict(checkpoint["network_state_dict"])
        print(f"Loaded model: {path}")
        return model
