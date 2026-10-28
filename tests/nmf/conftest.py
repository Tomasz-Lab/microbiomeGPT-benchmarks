from dataclasses import dataclass

import numpy as np
import pytest

from gut_former.nmf.nmf_core import NMF


@dataclass(frozen=True, eq=False, slots=True)
class SyntheticNMF:
    X: np.ndarray
    W: np.ndarray
    H: np.ndarray
    k: int


def make_synthetic_nmf(n_samples=180, n_features=30, n_components=3, seed=0):
    rng = np.random.default_rng(seed)

    # create signature components with random values
    block = n_features // n_components
    H = np.full((n_components, n_features), 0.01)
    for i in range(n_components):
        start = i * block
        stop = start + block
        H[i, start:stop] = rng.uniform(1, 3, size=block)

    # Dirichlet makes it easy to generate samples dominated by a single signature,
    # which makes them easier to reconstruct. It alwasy generate vector that sum to 1.
    # It tries to simulates proportions
    W = rng.dirichlet([0.2] * n_components, size=n_samples)
    # the uniform is need here to rescale the proportions to simulate the "real" abundance
    W = W * rng.uniform(5, 10, size=(n_samples, 1))

    X = W @ H

    for arr in (X, W, H):
        # in order to later don't be able to by mistake change arr values
        arr.setflags(write=False)

    return SyntheticNMF(X=X, W=W, H=H, k=n_components)


@pytest.fixture(scope="session")
def synthetic_nmf():
    return make_synthetic_nmf(n_samples=180, n_features=30, n_components=3)


@pytest.fixture(scope="session")
def nmf_data_factory():
    return make_synthetic_nmf


@pytest.fixture(scope="session")
def fitted_nmf_model(synthetic_nmf):
    return NMF(
        n_components=synthetic_nmf.k,
        n_runs=50,
        init="nndsvdar",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=2000,
        n_jobs=1,
        selection_metric="exp_var",
    ).fit(synthetic_nmf.X)
