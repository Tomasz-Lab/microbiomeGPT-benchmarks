from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

import gut_former.nmf.run_bicv as bicv
from gut_former.nmf.nmf_core import NMF


@dataclass(frozen=True, eq=False, slots=True)
class SyntheticNMF:
    X: np.ndarray
    W: np.ndarray
    H: np.ndarray
    k: int


@dataclass(frozen=True, eq=False, slots=True)
class SyntheticSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    W_train: np.ndarray
    W_val: np.ndarray
    H: np.ndarray
    k: int


def make_synthetic_nmf(n_samples=180, n_features=30, n_components=3, seed=0) -> SyntheticNMF:
    rng = np.random.default_rng(seed)

    # signatures: each component is active on its own block of features
    block = n_features // n_components
    H = np.full((n_components, n_features), 0.01)
    for i in range(n_components):
        start = i * block
        stop = start + block
        H[i, start:stop] = rng.uniform(1, 3, size=block)

    # Dirichlet gives proportions (rows sum to 1); alpha < 1 -> samples dominated
    # by one signature, which makes them easier to reconstruct
    W = rng.dirichlet([0.2] * n_components, size=n_samples)
    # rescale each sample to simulate different "real" abundances
    W = W * rng.uniform(5, 10, size=(n_samples, 1))

    X = W @ H

    # read-only, so no test can accidentally modify shared data
    for arr in (X, W, H):
        arr.setflags(write=False)

    return SyntheticNMF(X=X, W=W, H=H, k=n_components)


def split_synthetic_nmf(data: SyntheticNMF, val_fraction=0.2, seed=0) -> SyntheticSplit:
    # X and W are shuffled with the same permutation, so they stay aligned
    X_train, X_val, W_train, W_val = train_test_split(
        data.X, data.W, test_size=val_fraction, random_state=seed
    )
    for arr in (X_train, X_val, W_train, W_val):
        arr.setflags(write=False)

    return SyntheticSplit(
        X_train=X_train, X_val=X_val, W_train=W_train, W_val=W_val, H=data.H, k=data.k
    )


@pytest.fixture(scope="session")
def synthetic_nmf():
    return make_synthetic_nmf(n_samples=180, n_features=30, n_components=3)


@pytest.fixture(scope="session")
def synthetic_split(synthetic_nmf):
    return split_synthetic_nmf(synthetic_nmf, val_fraction=0.2)


@pytest.fixture(scope="session")
def nmf_data_factory():
    return make_synthetic_nmf


@pytest.fixture(scope="session")
def split_factory():
    return split_synthetic_nmf


@pytest.fixture(scope="session")
def nmf_params(synthetic_nmf):
    return dict(
        n_components=synthetic_nmf.k,
        n_runs=10,
        init="nndsvdar",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=2000,
        n_jobs=1,
        selection_metric="exp_var",
    )


@pytest.fixture
def nmf_model(nmf_params):
    return NMF(**nmf_params)


@pytest.fixture(scope="session")
def fitted_nmf_model(synthetic_nmf, nmf_params):
    return NMF(**nmf_params).fit(synthetic_nmf.X)


@pytest.fixture(scope="module")
def bicv_folds(synthetic_nmf):
    rng = np.random.default_rng(seed=0)
    matrix = bicv.shuffle_rows_and_cols(synthetic_nmf.X, rng)
    return {idx: fold for idx, fold in enumerate(bicv.make_folds(matrix, h=3), start=1)}
