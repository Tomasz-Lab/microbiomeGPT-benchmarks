from dataclasses import dataclass

import pytest
import torch

from gut_former.model import BacteriaModel


@dataclass(frozen=True, slots=True)
class ModelDims:
    num_pathways: int
    num_bacteria: int
    embedding_dim: int
    latent_size: int


@dataclass(frozen=True, eq=False, slots=True)
class ModelInputs:
    pathways: torch.Tensor
    taxonomy: torch.Tensor


DIMS = ModelDims(num_pathways=7, num_bacteria=5, embedding_dim=8, latent_size=4)
BATCH_SIZE = 4


def make_model(seed=0, dims=DIMS) -> BacteriaModel:
    # fork_rng restores the global RNG afterwards, so seeding here doesn't leak into other tests
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return BacteriaModel(
            num_pathways=dims.num_pathways,
            num_bacteria=dims.num_bacteria,
            embedding_dim=dims.embedding_dim,
            latent_size=dims.latent_size,
        )


def make_inputs(batch_size=BATCH_SIZE, dims=DIMS, seed=0) -> ModelInputs:
    gen = torch.Generator().manual_seed(seed)
    return ModelInputs(
        pathways=torch.rand(batch_size, dims.num_pathways, 1, generator=gen),
        taxonomy=torch.rand(batch_size, dims.num_bacteria, 1, generator=gen),
    )


@pytest.fixture(scope="session")
def dims():
    return DIMS


@pytest.fixture(scope="session")
def model_factory():
    return make_model


@pytest.fixture
def model():
    return make_model(seed=0).eval()


@pytest.fixture
def inputs():
    return make_inputs()
