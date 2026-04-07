"""
Tests for GaussianCountModel + integration with the recruiting pipeline.

All tests use synthetic dummy data by default (72-dim one-hot vectors matching
the real covariate schema) so no ICPSR download is required.

To run (dummy data only — no downloads needed):
    pytest tests/test_gaussian_count_model.py -v

ICPSR real-data tests run automatically when the data is found at any of:
    src/data/ICPSR_22140/    ← recommended: place data inside the package
    ICPSR_22140/             ← repo-root default (gitignored)
    $ICPSR_DATA_DIR          ← env var override

Optionally set DDPM_CHECKPOINT=/path/to/ddpm_HIV.pt to also test a pre-trained
diffusion model checkpoint.

Download dataset: https://www.icpsr.umich.edu/web/ICPSR/studies/22140
"""

from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.data.covariate_spec import COVARIATE_DIM, COVARIATE_GROUPS, validate_one_hot
from src.environment.recruiting_env import RecruitingEnv
from src.models.count_model.abstract_count_model import AbstractCountModel
from src.models.count_model.gaussian_count_model import GaussianCountModel
from src.models.covariate_model.abstract_covariate_model import AbstractCovariateModel
from src.models.covariate_model.ddpm_covariate_model import DDPMCovariateModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_one_hot_covariates(n: int, seed: int = 0) -> np.ndarray:
    """Generate n valid 72-dim one-hot covariate vectors.

    For each covariate group a random one-hot entry is set. The result passes
    validate_one_hot() and is compatible with all model inputs.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, COVARIATE_DIM), dtype=np.float64)
    for _, start, end in COVARIATE_GROUPS:
        choices = rng.integers(0, end - start, size=n)
        X[np.arange(n), start + choices] = 1.0
    assert validate_one_hot(X), "Generated covariates failed one-hot validation"
    return X


def make_synthetic_degrees(n: int, seed: int = 0, max_degree: int = 5) -> np.ndarray:
    """Generate n non-negative integer degree values (out-degree / # children)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, max_degree + 1, size=n).astype(np.float64)


# ---------------------------------------------------------------------------
# Minimal stub covariate model (no training needed)
# ---------------------------------------------------------------------------

class _DummyCovariateModel(AbstractCovariateModel):
    """Returns random valid one-hot covariates regardless of parent input.

    Makes integration tests fast and self-contained — no DDPM training.
    All abstract methods are implemented as no-ops or stubs.
    """

    def train(self, dataset, epochs=4000, batch_size=32,
              learning_rate=1e-3, seed=42, log_interval=100) -> dict:
        return {}

    def sample(self, parent_covariates: np.ndarray, seed: int = 42) -> np.ndarray:
        n = parent_covariates.shape[0]
        return make_one_hot_covariates(n, seed=seed)

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "_DummyCovariateModel":
        return cls()


# ===========================================================================
# 1. Unit tests — GaussianCountModel in isolation
# ===========================================================================

class TestGaussianCountModelUnit:
    """Focused unit tests with tiny synthetic datasets."""

    N_TRAIN = 80
    N_PRED = 10

    @pytest.fixture
    def trained_model(self) -> GaussianCountModel:
        X = make_one_hot_covariates(self.N_TRAIN, seed=1)
        y = make_synthetic_degrees(self.N_TRAIN, seed=1)
        model = GaussianCountModel(seed=42, max_train_size=50)
        model.fit(X, y)
        return model

    # --- fit / predict basics ---

    def test_fit_completes(self):
        X = make_one_hot_covariates(self.N_TRAIN)
        y = make_synthetic_degrees(self.N_TRAIN)
        model = GaussianCountModel(seed=0, max_train_size=50)
        model.fit(X, y)
        assert model._fitted

    def test_predict_output_shape(self, trained_model):
        X = make_one_hot_covariates(self.N_PRED, seed=99)
        allocs = np.full(self.N_PRED, 5, dtype=int)
        counts = trained_model.predict(X, allocs)
        assert counts.shape == (self.N_PRED,), f"expected ({self.N_PRED},), got {counts.shape}"

    def test_predict_nonnegative(self, trained_model):
        X = make_one_hot_covariates(self.N_PRED, seed=2)
        allocs = np.full(self.N_PRED, 5, dtype=int)
        counts = trained_model.predict(X, allocs)
        assert np.all(counts >= 0), "predict returned negative counts"

    def test_predict_integer_dtype(self, trained_model):
        X = make_one_hot_covariates(self.N_PRED, seed=3)
        allocs = np.full(self.N_PRED, 5, dtype=int)
        counts = trained_model.predict(X, allocs)
        assert np.issubdtype(counts.dtype, np.integer), \
            f"expected integer dtype, got {counts.dtype}"

    def test_predict_respects_allocation_cap(self, trained_model):
        X = make_one_hot_covariates(self.N_PRED, seed=4)
        allocs = np.arange(self.N_PRED, dtype=int)   # 0, 1, ..., N-1
        counts = trained_model.predict(X, allocs)
        assert np.all(counts <= allocs), \
            f"some count exceeded allocation: counts={counts}, allocs={allocs}"

    def test_predict_with_zero_allocations(self, trained_model):
        """When all allocations are 0, all counts must be 0."""
        X = make_one_hot_covariates(self.N_PRED, seed=5)
        allocs = np.zeros(self.N_PRED, dtype=int)
        counts = trained_model.predict(X, allocs)
        assert np.all(counts == 0), "non-zero counts when all allocations are 0"

    def test_predict_single_row(self, trained_model):
        X = make_one_hot_covariates(1, seed=6)
        allocs = np.array([3], dtype=int)
        counts = trained_model.predict(X, allocs)
        assert counts.shape == (1,)
        assert 0 <= counts[0] <= 3

    # --- input validation ---

    def test_fit_rejects_1d_covariates(self):
        model = GaussianCountModel()
        with pytest.raises(ValueError, match="2-D"):
            model.fit(np.zeros(72), np.zeros(1))

    def test_fit_rejects_mismatched_lengths(self):
        model = GaussianCountModel()
        with pytest.raises(ValueError, match="same number"):
            model.fit(np.zeros((10, 72)), np.zeros(8))

    def test_fit_rejects_negative_degrees(self):
        model = GaussianCountModel()
        y = np.full(10, -1.0)
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(np.zeros((10, 72)), y)

    def test_predict_before_fit_raises(self):
        model = GaussianCountModel()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.predict(make_one_hot_covariates(5), np.ones(5, dtype=int))

    def test_predict_rejects_shape_mismatch(self, trained_model):
        X = make_one_hot_covariates(5)
        allocs = np.ones(7, dtype=int)   # wrong length
        with pytest.raises(ValueError, match="shape"):
            trained_model.predict(X, allocs)

    # --- all-zero degree training ---

    def test_fit_all_zero_degrees(self):
        """Edge case: all nodes have 0 children (leaf nodes only)."""
        X = make_one_hot_covariates(50)
        y = np.zeros(50)
        model = GaussianCountModel(seed=0, max_train_size=50)
        model.fit(X, y)
        counts = model.predict(make_one_hot_covariates(10, seed=7), np.full(10, 3, dtype=int))
        assert np.all(counts == 0), "expected all-zero predictions for all-zero training degrees"

    # --- subsampling ---

    def test_subsampling_happens_when_large_dataset(self):
        """Verify that a large dataset is silently subsampled without error."""
        X = make_one_hot_covariates(300, seed=10)
        y = make_synthetic_degrees(300, seed=10)
        model = GaussianCountModel(seed=42, max_train_size=50)
        model.fit(X, y)  # should not raise
        assert model._fitted

    def test_no_subsampling_when_small_dataset(self):
        """Small dataset should train on all rows."""
        X = make_one_hot_covariates(30, seed=11)
        y = make_synthetic_degrees(30, seed=11)
        model = GaussianCountModel(seed=42, max_train_size=200)
        model.fit(X, y)
        assert model._fitted

    # --- save / load round-trip ---

    def test_save_load_roundtrip(self, trained_model):
        X = make_one_hot_covariates(self.N_PRED, seed=8)
        allocs = np.full(self.N_PRED, 4, dtype=int)
        counts_before = trained_model.predict(X, allocs)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            trained_model.save(path)
            loaded = GaussianCountModel.load(path)
        finally:
            os.unlink(path)

        assert loaded._fitted
        counts_after = loaded.predict(X, allocs)
        np.testing.assert_array_equal(
            counts_before, counts_after,
            err_msg="save/load changed predictions"
        )

    def test_save_load_preserves_hyperparams(self):
        model = GaussianCountModel(seed=7, alpha=0.5, n_restarts_optimizer=3, max_train_size=100)
        X = make_one_hot_covariates(60, seed=9)
        y = make_synthetic_degrees(60, seed=9)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            loaded = GaussianCountModel.load(path)
        finally:
            os.unlink(path)

        assert loaded.seed == 7
        assert loaded.alpha == 0.5
        assert loaded.n_restarts_optimizer == 3
        assert loaded.max_train_size == 100

    # --- AbstractCountModel interface ---

    def test_is_abstract_count_model_subclass(self):
        assert issubclass(GaussianCountModel, AbstractCountModel)


# ===========================================================================
# 2. Integration tests — GaussianCountModel inside RecruitingEnv
# ===========================================================================

class TestIntegrationWithDummyCovariateModel:
    """End-to-end tests wiring GaussianCountModel into RecruitingEnv.

    Uses _DummyCovariateModel (no training required) so tests run fast.
    """

    BUDGET = 20
    FRONTIER_SIZE = 5

    @pytest.fixture
    def env_with_models(self) -> RecruitingEnv:
        X_train = make_one_hot_covariates(80, seed=20)
        y_train = make_synthetic_degrees(80, seed=20)
        count_model = GaussianCountModel(seed=42, max_train_size=80)
        count_model.fit(X_train, y_train)

        covariate_model = _DummyCovariateModel()
        env = RecruitingEnv(
            covariate_model=covariate_model,
            count_model=count_model,
            initial_budget=self.BUDGET,
            discount_factor=0.9,
            max_rounds=10,
            seed=0,
        )
        return env

    @pytest.fixture
    def initial_frontier(self) -> np.ndarray:
        return make_one_hot_covariates(self.FRONTIER_SIZE, seed=30)

    def test_env_step_completes(self, env_with_models, initial_frontier):
        env = env_with_models
        state = env.reset(initial_frontier)
        action = np.ones(self.FRONTIER_SIZE, dtype=int)  # allocate 1 to each
        next_state, reward, done, info = env.step(action)
        assert reward >= 0
        assert "counts" in info
        assert info["counts"].shape == (self.FRONTIER_SIZE,)

    def test_env_counts_within_allocations(self, env_with_models, initial_frontier):
        env = env_with_models
        state = env.reset(initial_frontier)
        # Heterogeneous allocations
        action = np.array([3, 1, 0, 2, 4], dtype=int)
        next_state, reward, done, info = env.step(action)
        counts = info["counts"]
        assert np.all(counts >= 0), "negative counts"
        assert np.all(counts <= action), "counts exceeded allocations"

    def test_zero_allocation_produces_zero_counts(self, env_with_models, initial_frontier):
        """All-zero action means no budget spent, frontier next step must be empty."""
        env = env_with_models
        env.reset(initial_frontier)
        action = np.zeros(self.FRONTIER_SIZE, dtype=int)
        next_state, reward, done, info = env.step(action)
        assert reward == 0.0, "expected 0 reward for all-zero allocations"
        assert next_state.frontier_size == 0

    def test_full_episode_terminates(self, env_with_models, initial_frontier):
        env = env_with_models
        state = env.reset(initial_frontier)
        done = False
        rounds = 0
        while not done:
            budget = state.budget_remaining
            n = state.frontier_size
            if n == 0 or budget <= 0:
                break
            # Equal-split allocation
            per_person = max(1, budget // n)
            action = np.minimum(np.full(n, per_person, dtype=int), budget)
            # Ensure we don't overspend
            if action.sum() > budget:
                action = np.zeros(n, dtype=int)
                action[0] = min(budget, per_person)
            state, _, done, info = env.step(action)
            rounds += 1
        assert rounds > 0, "episode ended before any steps were taken"

    def test_cumulative_reward_nonneg(self, env_with_models, initial_frontier):
        env = env_with_models
        state = env.reset(initial_frontier)
        for _ in range(3):
            if state.frontier_size == 0 or state.budget_remaining <= 0:
                break
            n = state.frontier_size
            budget = state.budget_remaining
            action = np.minimum(np.ones(n, dtype=int), budget // max(n, 1))
            if action.sum() > budget:
                action = np.zeros(n, dtype=int)
            state, _, done, _ = env.step(action)
            if done:
                break
        assert env.cumulative_reward >= 0


# ===========================================================================
# 3. Integration tests — tiny trained DDPM covariate model
# ===========================================================================

class _TinyEdgePairDataset(Dataset):
    """Minimal dataset of (parent_cov || child_cov) pairs for fast DDPM training."""

    def __init__(self, n: int = 200, seed: int = 0):
        parents = make_one_hot_covariates(n, seed=seed)
        children = make_one_hot_covariates(n, seed=seed + 1)
        data = np.concatenate([parents, children], axis=1).astype(np.float32)
        self.data = torch.tensor(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class TestIntegrationWithTinyDDPM:
    """Integration tests using a lightly trained DDPM (fast but real)."""

    @pytest.fixture(scope="class")
    def tiny_ddpm(self):
        """Train a tiny DDPM for a few epochs — enough to confirm the pipeline works."""
        model = DDPMCovariateModel(hidden_dim=64, num_steps=10)
        dataset = _TinyEdgePairDataset(n=200, seed=0)
        model.train(dataset, epochs=3, batch_size=32, log_interval=999)
        return model

    @pytest.fixture(scope="class")
    def trained_count_model(self):
        X = make_one_hot_covariates(80, seed=40)
        y = make_synthetic_degrees(80, seed=40)
        m = GaussianCountModel(seed=42, max_train_size=80)
        m.fit(X, y)
        return m

    def test_ddpm_sample_shape(self, tiny_ddpm):
        parents = make_one_hot_covariates(5, seed=50)
        children = tiny_ddpm.sample(parents, seed=1)
        assert children.shape == (5, COVARIATE_DIM), \
            f"DDPM sample shape {children.shape} != (5, {COVARIATE_DIM})"

    def test_ddpm_sample_valid_one_hot(self, tiny_ddpm):
        parents = make_one_hot_covariates(10, seed=51)
        children = tiny_ddpm.sample(parents, seed=2)
        assert validate_one_hot(children), "DDPM output failed one-hot validation"

    def test_full_pipeline_one_step(self, tiny_ddpm, trained_count_model):
        """GaussianCountModel + DDPMCovariateModel + RecruitingEnv — one step."""
        env = RecruitingEnv(
            covariate_model=tiny_ddpm,
            count_model=trained_count_model,
            initial_budget=15,
            max_rounds=5,
            seed=0,
        )
        frontier = make_one_hot_covariates(4, seed=60)
        state = env.reset(frontier)
        action = np.array([2, 1, 3, 1], dtype=int)
        next_state, reward, done, info = env.step(action)

        # Smoke checks
        assert reward >= 0
        assert next_state.frontier_covariates.ndim == 2
        if next_state.frontier_size > 0:
            assert next_state.frontier_covariates.shape[1] == COVARIATE_DIM
        assert info["budget_spent"] == int(action.sum())
        assert 0 <= info["budget_spent"] <= 15

    def test_ddpm_save_load_then_predict(self, tiny_ddpm, trained_count_model):
        """Save DDPM to disk, reload it, then run a full env step."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            tiny_ddpm.save(path)
            loaded_ddpm = DDPMCovariateModel.load(path)
        finally:
            os.unlink(path)

        env = RecruitingEnv(
            covariate_model=loaded_ddpm,
            count_model=trained_count_model,
            initial_budget=10,
            max_rounds=3,
            seed=0,
        )
        frontier = make_one_hot_covariates(3, seed=70)
        state = env.reset(frontier)
        action = np.array([2, 1, 2], dtype=int)
        next_state, reward, done, info = env.step(action)
        assert reward >= 0


# ===========================================================================
# 4. (Optional) Real ICPSR data tests — requires --icpsr flag
# ===========================================================================

def _find_icpsr_data_dir() -> str | None:
    """Return path to ICPSR_22140 data directory, or None if not found.

    Checks (in order):
    1. ICPSR_DATA_DIR environment variable
    2. src/data/ICPSR_22140/ (data placed inside the package)
    3. ICPSR_22140/ at the repo root (gitignored default location)
    """
    env_dir = os.getenv("ICPSR_DATA_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(_REPO_ROOT, "src", "data", "ICPSR_22140"),
        os.path.join(_REPO_ROOT, "ICPSR_22140"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


_ICPSR_DIR = _find_icpsr_data_dir()


@pytest.mark.skipif(
    _ICPSR_DIR is None,
    reason=(
        "ICPSR 22140 data not found. Place it at src/data/ICPSR_22140/ or "
        "ICPSR_22140/ (repo root), or set ICPSR_DATA_DIR env var. "
        "Download from https://www.icpsr.umich.edu/web/ICPSR/studies/22140"
    ),
)
class TestWithRealICPSRData:
    """Tests that use the real ICPSR 22140 dataset and optionally a saved DDPM checkpoint.

    Data is auto-detected from:
    - ICPSR_DATA_DIR env var
    - src/data/ICPSR_22140/
    - ICPSR_22140/ (repo root, gitignored)

    Optional DDPM checkpoint:
        export DDPM_CHECKPOINT=/path/to/ddpm_HIV.pt

    Run:
        pytest tests/test_gaussian_count_model.py -v
    """

    @pytest.fixture(scope="class")
    def graph_data(self):
        from src.data.icpsr_loader import ICPSRGraphData
        return ICPSRGraphData(_ICPSR_DIR, std_name="HIV")

    @pytest.fixture(scope="class")
    def count_model_icpsr(self, graph_data):
        common_nodes = sorted(
            set(graph_data.covariates) & set(graph_data.node_degrees)
        )
        X = np.array([graph_data.covariates[n] for n in common_nodes])
        y = np.array([graph_data.node_degrees[n] for n in common_nodes], dtype=np.float64)
        model = GaussianCountModel(seed=42, max_train_size=2000)
        model.fit(X, y)
        return model

    def test_icpsr_fit_completes(self, count_model_icpsr):
        assert count_model_icpsr._fitted

    def test_icpsr_predict_on_real_frontier(self, graph_data, count_model_icpsr):
        frontier_nodes = list(graph_data.covariates.keys())[:20]
        X = np.array([graph_data.covariates[n] for n in frontier_nodes])
        allocs = np.full(len(frontier_nodes), 5, dtype=int)
        counts = count_model_icpsr.predict(X, allocs)
        assert counts.shape == (len(frontier_nodes),)
        assert np.all(counts >= 0)
        assert np.all(counts <= allocs)

    def test_icpsr_env_episode(self, graph_data, count_model_icpsr):
        covariate_model = _DummyCovariateModel()
        env = RecruitingEnv(
            covariate_model=covariate_model,
            count_model=count_model_icpsr,
            initial_budget=50,
            max_rounds=10,
            seed=0,
        )
        frontier = graph_data.sample_initial_frontier(n=10, seed=0)
        state = env.reset(frontier)
        done = False
        while not done:
            n = state.frontier_size
            budget = state.budget_remaining
            if n == 0 or budget <= 0:
                break
            per = max(1, budget // n)
            action = np.minimum(np.full(n, per, dtype=int), budget)
            if action.sum() > budget:
                action = np.zeros(n, dtype=int)
            state, _, done, _ = env.step(action)
        assert env.cumulative_reward >= 0

    @pytest.mark.skipif(
        not os.getenv("DDPM_CHECKPOINT"),
        reason="Set DDPM_CHECKPOINT to test with a real trained diffusion model",
    )
    def test_icpsr_with_real_ddpm(self, graph_data, count_model_icpsr):
        ckpt = os.environ["DDPM_CHECKPOINT"]
        covariate_model = DDPMCovariateModel.load(ckpt)
        env = RecruitingEnv(
            covariate_model=covariate_model,
            count_model=count_model_icpsr,
            initial_budget=30,
            max_rounds=5,
            seed=0,
        )
        frontier = graph_data.sample_initial_frontier(n=5, seed=1)
        state = env.reset(frontier)
        action = np.ones(state.frontier_size, dtype=int)
        if action.sum() > state.budget_remaining:
            action = np.zeros(state.frontier_size, dtype=int)
        next_state, reward, done, info = env.step(action)
        assert reward >= 0
