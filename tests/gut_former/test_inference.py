import pandas as pd
import pytest
from hydra.core.hydra_config import HydraConfig

from gut_former.run_inference import run_inference
from gut_former.run_training import run_training

pytestmark = pytest.mark.integration


def test_inference_writes_predictions_from_trained_checkpoint(
    cfg_train, cfg_infer, tmp_path, tiny_dataset
):
    # Given
    HydraConfig().set_config(cfg_train)
    run_training(cfg_train)
    cfg_infer.task.checkpoint_path = str(tmp_path / "checkpoints" / "best.ckpt")
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
