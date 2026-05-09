"""
PyTorch Dataset for parent-child covariate edge pairs.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EdgePairDataset(Dataset):
    """Dataset of (parent_cov, child_cov) pairs for diffusion model training.

    Each sample is a 144-dim tensor: [parent_cov(72) || child_cov(72)],
    matching the format expected by the reference diffusion model.

    Args:
        edge_pairs: List of (parent_cov, child_cov) tuples, each a 72-dim array.
    """

    def __init__(self, edge_pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        data = []
        for parent, child in edge_pairs:
            data.append(np.concatenate([parent, child]))
        self.data = torch.tensor(np.array(data), dtype=torch.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
