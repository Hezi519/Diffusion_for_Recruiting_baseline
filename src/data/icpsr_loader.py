"""
ICPSR 22140 graph data loader for diffusion model training.

Extracts parent-child edge pairs and node covariates from the ICPSR 22140
dataset for training the covariate diffusion model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from src.data.icpsr_processor import ICPSRProcessor


class ICPSRGraphData:
    """Loads ICPSR data and extracts parent-child edge pairs for diffusion training.

    Args:
        data_dir: Path to the ICPSR_22140 directory containing DS0001/, DS0002/, DS0003/.
        std_name: Which disease network to use (e.g. "HIV", "Gonorrhea").
        pickle_cache: Path for caching processed data. If None, uses data_dir/processed.pkl.
    """

    def __init__(
        self,
        data_dir: str,
        std_name: str = "HIV",
        pickle_cache: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.std_name = std_name

        tsv_file1 = str(self.data_dir / "DS0001" / "22140-0001-Data.tsv")
        tsv_file2 = str(self.data_dir / "DS0002" / "22140-0002-Data.tsv")
        tsv_file3 = str(self.data_dir / "DS0003" / "22140-0003-Data.tsv")
        if pickle_cache is None:
            pickle_cache = str(self.data_dir / "processed.pkl")

        self._processor = ICPSRProcessor(
            tsv_file1, tsv_file2, tsv_file3, pickle_cache
        )

        merged = self._processor.merged_datasets[self.std_name]
        self._covariates: dict[int, np.ndarray] = merged[0]
        self._statuses: dict[int, int] = merged[1]
        self._graph: nx.Graph = merged[2]
        self._digraph: nx.DiGraph = merged[3]
        self._graph_roots: np.ndarray = merged[4]
        self._digraph_roots: np.ndarray = merged[5]

        # Remove self-loops (as done in the reference explanation.md)
        self._graph.remove_edges_from(nx.selfloop_edges(self._graph))
        self._digraph.remove_edges_from(nx.selfloop_edges(self._digraph))

        # Build edge pairs: (parent_cov, child_cov) from directed edges
        self._edge_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for u, v in self._digraph.edges():
            if u in self._covariates and v in self._covariates:
                self._edge_pairs.append(
                    (self._covariates[u], self._covariates[v])
                )

    @property
    def edge_pairs(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """List of (parent_cov_72, child_cov_72) pairs from directed graph edges."""
        return self._edge_pairs

    @property
    def graph(self) -> nx.Graph:
        """The undirected contact graph."""
        return self._graph

    @property
    def digraph(self) -> nx.DiGraph:
        """The directed graph (edge direction = recruitment direction)."""
        return self._digraph

    @property
    def covariates(self) -> dict[int, np.ndarray]:
        """Node ID -> 72-dim covariate vector."""
        return self._covariates

    @property
    def statuses(self) -> dict[int, int]:
        """Node ID -> disease status (0 or 1)."""
        return self._statuses

    @property
    def node_degrees(self) -> dict[int, int]:
        """Node ID -> out-degree in the directed graph (number of children recruited)."""
        return {node: self._digraph.out_degree(node) for node in self._digraph.nodes()}

    @property
    def roots(self) -> np.ndarray:
        """Root nodes (seeds) in the directed graph."""
        return self._digraph_roots

    def train_test_split(
        self,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
        """Split edge pairs into train/test sets.

        Args:
            test_fraction: Fraction of edges for the test set.
            seed: Random seed for reproducibility.

        Returns:
            (train_pairs, test_pairs) where each is a list of (parent_cov, child_cov).
        """
        rng = np.random.default_rng(seed)
        n = len(self._edge_pairs)
        indices = rng.permutation(n)
        split = int(n * (1 - test_fraction))
        train_idx = indices[:split]
        test_idx = indices[split:]
        train_pairs = [self._edge_pairs[i] for i in train_idx]
        test_pairs = [self._edge_pairs[i] for i in test_idx]
        return train_pairs, test_pairs

    def train_test_node_split(
        self,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> tuple[set[int], set[int]]:
        """Split nodes into train/test sets for count model fitting.

        Uses a node-level split (not edge-level) so that a node's ground-truth
        out-degree is either fully in train or fully held out, preventing the
        count model from memorising test-node recruitment counts.

        Args:
            test_fraction: Fraction of nodes to hold out.
            seed: Random seed (use the same seed as train_test_split for
                  consistency).

        Returns:
            (train_nodes, test_nodes) as sets of integer node IDs.
        """
        rng = np.random.default_rng(seed)
        all_nodes = sorted(self._covariates.keys())
        n = len(all_nodes)
        shuffled = rng.permutation(n)
        split = int(n * (1 - test_fraction))
        train_nodes = {all_nodes[i] for i in shuffled[:split]}
        test_nodes = {all_nodes[i] for i in shuffled[split:]}
        return train_nodes, test_nodes

    def sample_initial_frontier(
        self,
        n: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Sample n covariate vectors to use as an initial frontier.

        Samples from the root nodes of the directed graph. If fewer roots
        than n, samples from all nodes.

        Args:
            n: Number of frontier individuals to sample.
            seed: Random seed.

        Returns:
            Array of shape (n, 72).
        """
        rng = np.random.default_rng(seed)
        candidates = list(self._digraph_roots) if len(self._digraph_roots) >= n else list(self._covariates.keys())
        chosen = rng.choice(candidates, size=n, replace=len(candidates) < n)
        return np.array([self._covariates[node] for node in chosen])
