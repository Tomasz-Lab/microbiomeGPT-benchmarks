import logging
import os

import pytest
import torch
from hydra.core.hydra_config import HydraConfig

from gut_former.run_training import run_training

pytestmark = pytest.mark.integration


def test_train_saves_best_and_last_checkpoints(cfg_train, tmp_path):
    # Given
    HydraConfig().set_config(cfg_train)

    # When
    run_training(cfg_train)

    # Then
    files = os.listdir(tmp_path / "checkpoints")
    assert sorted(files) == ["best.ckpt", "last.ckpt"]


def test_best_checkpoint_has_no_training_state(cfg_train, tmp_path):
    # Given
    HydraConfig().set_config(cfg_train)
    run_training(cfg_train)

    # When
    best = torch.load(tmp_path / "checkpoints" / "best.ckpt", map_location="cpu", weights_only=True)

    # Then
    assert "state_dict" in best
    assert "optimizer_states" not in best
    assert "lr_schedulers" not in best
    assert "callbacks" not in best


def test_last_checkpoint_has_training_state(cfg_train, tmp_path):
    # Given
    HydraConfig().set_config(cfg_train)
    run_training(cfg_train)

    # When
    last = torch.load(tmp_path / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True)

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
def test_checkpoint_holds_only_expected_content(cfg_train, tmp_path, filename):
    # Given
    HydraConfig().set_config(cfg_train)
    run_training(cfg_train)

    # When
    checkpoint = torch.load(
        tmp_path / "checkpoints" / filename, map_location="cpu", weights_only=False
    )
    non_model_weights = [key for key in checkpoint["state_dict"] if not key.startswith("model.")]

    # Then
    assert _non_plain_values(checkpoint) == []
    assert set(checkpoint["hyper_parameters"]) == {"model_schema", "loss", "optimizer"}
    assert non_model_weights == []


@pytest.mark.parametrize("filename", ["best.ckpt", "last.ckpt"])
def test_checkpoint_contains_model_schema(cfg_train, tmp_path, filename):
    # Given
    HydraConfig().set_config(cfg_train)
    run_training(cfg_train)

    # When
    checkpoint = torch.load(
        tmp_path / "checkpoints" / filename, map_location="cpu", weights_only=True
    )

    # Then
    assert "model_schema" in checkpoint["hyper_parameters"]


def test_train_restores_from_resume_from_checkpoint(cfg_train, tmp_path, caplog):
    # Given
    HydraConfig().set_config(cfg_train)
    run_training(cfg_train)
    checkpoint_path = str(tmp_path / "checkpoints" / "last.ckpt")
    cfg_train.task.resume_from = checkpoint_path
    cfg_train.task.verbose = True
    cfg_train.trainer.max_epochs = 2

    # When
    with caplog.at_level(logging.INFO):
        run_training(cfg_train)

    # Then
    assert f"Restoring states from the checkpoint path at {checkpoint_path}" in caplog.text
