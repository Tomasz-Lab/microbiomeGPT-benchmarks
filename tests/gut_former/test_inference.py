import logging

import pandas as pd
import pytest
import torch
from hydra.core.hydra_config import HydraConfig

from gut_former.run_inference import run_inference

pytestmark = pytest.mark.integration


def test_inference_writes_predictions_from_trained_checkpoint(
    cfg_infer, trained_run_dir, tmp_path, tiny_dataset
):
    # Given
    cfg_infer.task.checkpoint_path = str(trained_run_dir / "checkpoints" / "best.ckpt")
    HydraConfig().set_config(cfg_infer)

    # When
    run_inference(cfg_infer)

    # Then
    for modality in ("taxonomy", "pathways"):
        inputs = pd.read_csv(tiny_dataset.dir / f"{modality}_{tiny_dataset.name}.csv", index_col=0)
        predictions = pd.read_csv(
            tmp_path / f"pred_{modality}_{tiny_dataset.name}.csv", index_col=0
        )
        assert sorted(predictions.index) == sorted(inputs.index)
        assert list(predictions.columns) == list(inputs.columns)
        assert predictions.notna().all().all()
    latent = pd.read_csv(tmp_path / f"latent_{tiny_dataset.name}.csv", index_col=0)
    assert sorted(latent.index) == sorted(inputs.index)


def test_inference_without_checkpoint_writes_no_predictions(cfg_infer, tmp_path):
    # Given
    HydraConfig().set_config(cfg_infer)

    # When / Then
    with pytest.raises(ValueError, match="checkpoint"):
        run_inference(cfg_infer)
    assert list(tmp_path.glob("*.csv")) == []


def test_inference_without_prediction_writer_fails_before_loading(cfg_infer, tmp_path):
    # Given: a trainer override that drops the PredictionWriter
    cfg_infer.task.checkpoint_path = str(tmp_path / "missing.ckpt")
    cfg_infer.trainer.callbacks = [{"_target_": "gut_former.callbacks.RunLogger"}]
    HydraConfig().set_config(cfg_infer)

    # When / Then
    with pytest.raises(ValueError, match="BasePredictionWriter"):
        run_inference(cfg_infer)
    assert list(tmp_path.glob("*.csv")) == []


def test_inference_refuses_checkpoint_with_disallowed_target(
    cfg_infer, trained_run_dir, tmp_path, caplog
):
    # Given: a copy of the trained checkpoint whose loss config was replaced with an arbitrary call
    checkpoint_path = trained_run_dir / "checkpoints" / "best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["hyper_parameters"]["loss"] = {"_target_": "os.system", "command": "echo pwned"}
    tampered_path = tmp_path / "tampered.ckpt"
    torch.save(checkpoint, tampered_path)
    cfg_infer.task.checkpoint_path = str(tampered_path)
    HydraConfig().set_config(cfg_infer)

    # When / Then
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="os.system"):
        run_inference(cfg_infer)
    assert "Model not loaded" in caplog.text
    assert list(tmp_path.glob("pred_*.csv")) == []
