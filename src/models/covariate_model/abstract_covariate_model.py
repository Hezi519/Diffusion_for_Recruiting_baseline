"""
Abstract interface for covariate generation models.

Any model that generates child covariates from parent covariates
(DDPM, flow matching, VAE, etc.) can be swapped in by implementing
this interface. The environment and training scripts only interact
with models through this abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset


class AbstractCovariateModel(ABC):
    """Interface for covariate generation models.

    Implementations must support:
    - Training from a dataset of (parent, child) covariate pairs
    - Sampling child covariates given parent covariates
    - Saving/loading model checkpoints
    """

    @abstractmethod
    def train(
        self,
        dataset: Dataset,
        epochs: int = 4000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 42,
        log_interval: int = 100,
    ) -> dict:
        """Train the model on a dataset of (parent, child) covariate pairs.

        Args:
            dataset: PyTorch Dataset where each sample is a 144-dim tensor
                     [parent_cov(72) || child_cov(72)].
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Optimizer learning rate.
            seed: Random seed for reproducibility.
            log_interval: Print loss every this many epochs.

        Returns:
            Dictionary of training metrics (e.g. {"final_loss": float}).
        """

    @abstractmethod
    def sample(
        self,
        parent_covariates: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate one child covariate vector per parent.

        Args:
            parent_covariates: (n, 72) array of parent covariate vectors.
            seed: Random seed for reproducibility.

        Returns:
            (n, 72) array of valid one-hot encoded child covariates.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint to disk.

        Args:
            path: File path for the checkpoint.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: str = "auto") -> AbstractCovariateModel:
        """Load model checkpoint from disk.

        Args:
            path: File path to the checkpoint.
            device: Device to load model onto ("auto", "cuda", "cpu").

        Returns:
            Loaded model instance.
        """
