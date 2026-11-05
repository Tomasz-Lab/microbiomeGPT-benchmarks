import pytest
import torch

from gut_former.model import BacteriaModel
from gut_former.schema import DataSchema


@pytest.fixture(scope="session")
def small_model_config():
    return {"num_pathways": 7, "num_bacteria": 5, "embedding_dim": 8, "latent_size": 4}


@pytest.fixture(scope="session")
def small_training_config():
    return {"batch_size": 4}


@pytest.fixture(scope="session")
def small_data_schema(small_model_config):
    return DataSchema(
        {
            "pathways": tuple(f"pathway_{i}" for i in range(small_model_config["num_pathways"])),
            "taxonomy": tuple(f"bacterium_{i}" for i in range(small_model_config["num_bacteria"])),
        }
    )


@pytest.fixture(scope="session")
def small_model(small_model_config, small_data_schema):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return BacteriaModel(data_schema=small_data_schema, **small_model_config).eval()


@pytest.fixture
def small_model_same_seed(small_model_config, small_data_schema):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return BacteriaModel(data_schema=small_data_schema, **small_model_config).eval()


@pytest.fixture
def small_model_other_seed(small_model_config, small_data_schema):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return BacteriaModel(data_schema=small_data_schema, **small_model_config).eval()


@pytest.fixture(scope="session")
def small_model_inputs(small_model_config, small_training_config):
    batch_size = small_training_config["batch_size"]
    gen = torch.Generator().manual_seed(0)
    pathways = torch.rand(batch_size, small_model_config["num_pathways"], 1, generator=gen)
    taxonomy = torch.rand(batch_size, small_model_config["num_bacteria"], 1, generator=gen)
    return pathways, taxonomy
