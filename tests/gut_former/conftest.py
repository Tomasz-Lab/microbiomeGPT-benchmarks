import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from gut_former.run_training import run_training
from settings.hydra import CONFIG_NAME, CONFIG_PATH, VERSION_BASE


@dataclass(frozen=True)
class TinyDataset:
    dir: Path
    name: str


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> TinyDataset:
    """Taxonomy, pathways and metadata CSVs with 20 samples, 6 taxa and 5 pathways.

    Rows sum to 100 (taxonomy, MetaPhlAn percent) and 1 (pathways, HUMAnN relab), as the
    `taxonomy_metaphlan` and `pathways_humann` modalities require.
    """
    dataset = TinyDataset(dir=tmp_path_factory.mktemp("data"), name="tiny")
    rng = np.random.default_rng(0)
    index = pd.Index([f"sample_{i}" for i in range(20)], name="sample")
    taxonomy = pd.DataFrame(
        rng.dirichlet(np.ones(6), 20) * 100,
        index=index,
        columns=[
            f"k__Bacteria|p__P|c__C|o__O|f__F|g__Genus{i}|s__Genus{i}_species" for i in range(6)
        ],
    )
    pathways = pd.DataFrame(
        rng.dirichlet(np.ones(5), 20),
        index=index,
        columns=[f"PWY-{i}: pathway {i}" for i in range(5)],
    )
    metadata = pd.DataFrame({"study_condition": ["control"] * 20}, index=index)
    taxonomy.to_csv(dataset.dir / f"taxonomy_{dataset.name}.csv")
    pathways.to_csv(dataset.dir / f"pathways_{dataset.name}.csv")
    metadata.to_csv(dataset.dir / f"metadata_{dataset.name}.csv")
    return dataset


def point_modataset_at(cfg: DictConfig, dataset: TinyDataset) -> None:
    """Read the dataset recipe's files from the tiny dataset instead of the real data."""
    cfg.modataset.name = dataset.name
    modalities = cfg.modataset.modalities
    modalities.taxonomy_metaphlan.path = str(dataset.dir / f"taxonomy_{dataset.name}.csv")
    modalities.pathways_humann.path = str(dataset.dir / f"pathways_{dataset.name}.csv")
    cfg.modataset.metadata.path = str(dataset.dir / f"metadata_{dataset.name}.csv")


@pytest.fixture(scope="session")
def cfg_train_global(tiny_dataset) -> DictConfig:
    """The real training experiment config; overrides only make the run small and fast."""
    # TODO: rename the experiment in the repository cleanup issue
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=VERSION_BASE):
        cfg = compose(
            config_name=CONFIG_NAME,
            return_hydra_config=True,
            overrides=["experiment=gut_former_training", "model=transformer_autoencoder"],
        )

    with open_dict(cfg):
        cfg.task.verbose = False
        point_modataset_at(cfg, tiny_dataset)
        cfg.datamodule.batch_size = 8
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
        point_modataset_at(cfg, tiny_dataset)
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
