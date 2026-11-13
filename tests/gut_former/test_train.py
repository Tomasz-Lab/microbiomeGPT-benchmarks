import copy
import hashlib
import logging
import os
import warnings
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict, read_write

from gut_former.run_training import run_training
from gut_former.utils.data_state import file_sha256

pytestmark = pytest.mark.integration


def test_train_saves_best_and_last_checkpoints(trained_run_dir):
    # When
    files = os.listdir(trained_run_dir / "checkpoints")

    # Then
    assert sorted(files) == ["best.ckpt", "last.ckpt"]


def test_best_checkpoint_has_no_training_state(trained_run_dir):
    # When
    best = torch.load(
        trained_run_dir / "checkpoints" / "best.ckpt", map_location="cpu", weights_only=True
    )

    # Then
    assert "state_dict" in best
    assert "optimizer_states" not in best
    assert "lr_schedulers" not in best
    assert "callbacks" not in best


def test_last_checkpoint_has_training_state(trained_run_dir):
    # When
    last = torch.load(
        trained_run_dir / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True
    )

    # Then
    assert "state_dict" in last
    assert last["optimizer_states"][0]["state"]
    assert "lr_schedulers" in last
    assert last["callbacks"]
    assert "loops" in last
    assert last["global_step"] > 0


PLAIN_TYPES = (str, int, float, bool, type(None), torch.Tensor)


def _non_plain_values(obj, path: str = "checkpoint") -> list[str]:
    """Paths of every value that is not a tensor, a plain type or a dict/list/tuple of them."""
    if isinstance(obj, dict):
        return [
            found
            for key, value in obj.items()
            for found in _non_plain_values(key, f"{path} key {key!r}")
            + _non_plain_values(value, f"{path}[{key!r}]")
        ]
    if isinstance(obj, list | tuple):
        return [
            found
            for i, value in enumerate(obj)
            for found in _non_plain_values(value, f"{path}[{i}]")
        ]
    if isinstance(obj, PLAIN_TYPES):
        return []
    return [f"{path}: {type(obj).__module__}.{type(obj).__qualname__}"]


@pytest.mark.parametrize("filename", ["best.ckpt", "last.ckpt"])
def test_checkpoint_holds_only_expected_content(trained_run_dir, filename):
    # When
    checkpoint = torch.load(
        trained_run_dir / "checkpoints" / filename, map_location="cpu", weights_only=False
    )
    non_model_weights = [key for key in checkpoint["state_dict"] if not key.startswith("model.")]

    # Then
    assert _non_plain_values(checkpoint) == []
    assert set(checkpoint["hyper_parameters"]) == {"model_schema", "loss", "optimizer"}
    assert non_model_weights == []


def test_train_restores_from_resume_from_checkpoint(cfg_train, trained_run_dir, tmp_path, caplog):
    # Given: cfg_train writes to tmp_path, a new output dir like every Hydra run gets
    HydraConfig().set_config(cfg_train)
    cfg_train.task.resume_from = str(trained_run_dir / "checkpoints" / "last.ckpt")
    cfg_train.task.verbose = True
    cfg_train.trainer.max_epochs = 2

    # When
    with caplog.at_level(logging.INFO):
        run_training(cfg_train)

    # Then
    copied_path = os.path.realpath(tmp_path / "checkpoints" / "last.ckpt")
    assert f"Restoring states from the checkpoint path at {copied_path}" in caplog.text


def test_best_epoch_and_checkpoints_are_logged(cfg_train, tmp_path):
    # Given
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 3
        cfg_train.trainer.logger = [
            {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": str(tmp_path),
                "name": "csvlogger",
                "version": "",
            }
        ]
    HydraConfig().set_config(cfg_train)
    # Like @hydra.main: the task config has no hydra node (loggers get it resolved)
    task_cfg = copy.deepcopy(cfg_train)
    with open_dict(task_cfg):
        del task_cfg["hydra"]

    # When
    run_training(task_cfg)

    # Then
    metrics = pd.read_csv(tmp_path / "csvlogger" / "metrics.csv")
    # Every epoch: min of val_loss so far and the epoch it was reached
    val = metrics.dropna(subset=["val_loss"]).set_index("epoch")
    assert val["val_loss_best"].tolist() == pytest.approx(val["val_loss"].cummin().tolist())
    expected_epochs = [val["val_loss"].loc[:epoch].idxmin() for epoch in val.index]
    assert val["val_loss_best_epoch"].tolist() == expected_epochs

    hparams = yaml.safe_load((tmp_path / "csvlogger" / "hparams.yaml").read_text())
    assert set(hparams["checkpoints"]) == {"best", "last"}
    for name, entry in hparams["checkpoints"].items():
        assert Path(entry["path"]) == (tmp_path / "checkpoints" / f"{name}.ckpt").resolve()
        assert entry["sha256"] == file_sha256(entry["path"])


def test_resume_into_the_resumed_run_folder_fails(cfg_train, trained_run_dir):
    # Given
    with read_write(cfg_train.hydra):
        cfg_train.hydra.runtime.output_dir = str(trained_run_dir)
    HydraConfig().set_config(cfg_train)
    cfg_train.task.resume_from = str(trained_run_dir / "checkpoints" / "last.ckpt")

    # When / Then
    with pytest.raises(ValueError, match="resume in a new output folder"):
        run_training(cfg_train)


def checkpoint_hashes(checkpoint_dir):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoint_dir.iterdir()
    }


def test_resume_keeps_best_score_and_leaves_resumed_run_untouched(
    cfg_train, trained_run_dir, tmp_path
):
    # Given: cfg_train writes to tmp_path, a new output dir like every Hydra run gets
    first_run_dir = trained_run_dir
    second_run_dir = tmp_path
    first_run_hashes = checkpoint_hashes(first_run_dir / "checkpoints")
    HydraConfig().set_config(cfg_train)
    cfg_train.task.resume_from = str(first_run_dir / "checkpoints" / "last.ckpt")
    cfg_train.trainer.max_epochs = 2

    # When
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_training(cfg_train)

    # Then
    assert not [w for w in caught if "dirpath has changed" in str(w.message)]
    assert checkpoint_hashes(first_run_dir / "checkpoints") == first_run_hashes
    assert sorted(os.listdir(second_run_dir / "checkpoints")) == ["best.ckpt", "last.ckpt"]
    checkpoint = torch.load(
        second_run_dir / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False
    )
    assert checkpoint["epoch"] == 1
    assert str(first_run_dir) not in str(checkpoint["callbacks"])
