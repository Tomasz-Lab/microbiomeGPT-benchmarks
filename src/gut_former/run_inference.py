import logging

import hydra
from lightning.pytorch.callbacks import BasePredictionWriter
from omegaconf import DictConfig

from gut_former.system import (
    build_model,
    is_lightning_checkpoint,
    load_model_state_dict,
    load_system,
    verify_checkpoint_schema,
)
from gut_former.utils.log_config import setup_logging

log = logging.getLogger(__name__)


def run_inference(cfg: DictConfig, **_):
    # Setting up project
    setup_logging(logging.INFO if cfg.task.verbose else logging.WARNING)

    log.info("Starting inference script..")

    if cfg.task.checkpoint_path is None:
        raise ValueError(
            "Inference requires a trained model checkpoint; set checkpoint_path in config."
        )

    # Loading & Preparing Data
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("predict")
    data_schema = datamodule.data_schema()

    # Recreating Model
    log.info("Recreating model from checkpoint: %s", cfg.task.checkpoint_path)
    verify_checkpoint_schema(cfg.task.checkpoint_path, data_schema)
    if is_lightning_checkpoint(cfg.task.checkpoint_path):
        system = load_system(
            cfg.task.checkpoint_path, system_class=hydra.utils.get_class(cfg.system._target_)
        )
    else:
        log.warning(
            "Legacy checkpoint: rebuilding the model from cfg.model and the inference data."
        )
        model = build_model(cfg.model, data_schema)
        model.load_state_dict(load_model_state_dict(cfg.task.checkpoint_path))
        system = hydra.utils.instantiate(cfg.system)(model=model)

    # Inference
    trainer = hydra.utils.instantiate(cfg.trainer)
    if not any(isinstance(callback, BasePredictionWriter) for callback in trainer.callbacks):
        log.warning(
            "No prediction writer is configured in trainer.callbacks; nothing will be saved."
        )

    # Outputs are written by the PredictionWriter callback (conf/trainer/inference.yaml)
    trainer.predict(system, datamodule=datamodule, return_predictions=False)
