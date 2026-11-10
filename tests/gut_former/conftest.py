import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from gut_former.model import BacteriaModel
from gut_former.run_training import run_training
from gut_former.schema import DataSchema
from settings.hydra import CONFIG_NAME, CONFIG_PATH, VERSION_BASE


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


@dataclass(frozen=True)
class TinyDataset:
    dir: Path
    name: str


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> TinyDataset:
    """Taxonomy and pathways CSVs with 20 samples, 6 taxa and 5 pathways."""
    dataset = TinyDataset(dir=tmp_path_factory.mktemp("data"), name="tiny")
    rng = np.random.default_rng(0)
    index = pd.Index([f"sample_{i}" for i in range(20)], name="sample")
    taxonomy = pd.DataFrame(
        rng.random((20, 6)), index=index, columns=[f"taxon_{i}" for i in range(6)]
    )
    pathways = pd.DataFrame(
        rng.random((20, 5)), index=index, columns=[f"pathway_{i}" for i in range(5)]
    )
    taxonomy.to_csv(dataset.dir / f"taxonomy_{dataset.name}.csv")
    pathways.to_csv(dataset.dir / f"pathways_{dataset.name}.csv")
    return dataset


@pytest.fixture(scope="session")
def cfg_train_global(tiny_dataset) -> DictConfig:
    """The real gut_former_training config; overrides only make the run small and fast."""
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=VERSION_BASE):
        cfg = compose(
            config_name=CONFIG_NAME,
            return_hydra_config=True,
            overrides=["experiment=gut_former_training"],
        )

    with open_dict(cfg):
        cfg.task.verbose = False
        cfg.datamodule.data_dir = str(tiny_dataset.dir)
        cfg.datamodule.name = tiny_dataset.name
        cfg.datamodule.batch_size = 8
        cfg.model.embedding_dim = 8
        cfg.model.latent_size = 4
        cfg.trainer.max_epochs = 1
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.enable_progress_bar = False
        cfg.trainer.logger = False

    return cfg


@pytest.fixture
def cfg_train(cfg_train_global, tmp_path) -> DictConfig:
    """A per-test copy whose run outputs (checkpoints, logs) go to tmp_path."""
    cfg = copy.deepcopy(cfg_train_global)

    with open_dict(cfg):
        # Read by ${hydra:runtime.output_dir} once the test calls HydraConfig().set_config(cfg)
        cfg.hydra.runtime.output_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="session")
def trained_run_dir(cfg_train_global, tmp_path_factory) -> Path:
    """Output folder of one tiny training run, shared by every test that reads its checkpoints.

    Training once keeps the integration suite fast. Tests must only read this folder; any test
    that trains again or writes files uses its own tmp_path.
    """
    run_dir = tmp_path_factory.mktemp("trained_run")
    cfg = copy.deepcopy(cfg_train_global)
    with open_dict(cfg):
        cfg.hydra.runtime.output_dir = str(run_dir)
    HydraConfig().set_config(cfg)

    run_training(cfg)

    GlobalHydra.instance().clear()
    return run_dir


@pytest.fixture(scope="session")
def cfg_infer_global(tiny_dataset) -> DictConfig:
    """The real gut_former_inference config; overrides only point it at the tiny data on CPU."""
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=VERSION_BASE):
        cfg = compose(
            config_name=CONFIG_NAME,
            return_hydra_config=True,
            overrides=["experiment=gut_former_inference"],
        )

    with open_dict(cfg):
        cfg.task.verbose = False
        cfg.dataset.name = tiny_dataset.name
        cfg.datamodule.data_dir = str(tiny_dataset.dir)
        cfg.trainer.accelerator = "cpu"

    return cfg


@pytest.fixture
def cfg_infer(cfg_infer_global, tmp_path) -> DictConfig:
    """A per-test copy whose predictions go to tmp_path."""
    cfg = copy.deepcopy(cfg_infer_global)

    with open_dict(cfg):
        # Read by ${hydra:runtime.output_dir} once the test calls HydraConfig().set_config(cfg)
        cfg.hydra.runtime.output_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
