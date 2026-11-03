import logging

import hydra
from lightning.pytorch.callbacks import BasePredictionWriter
from omegaconf import DictConfig

from gut_former.lit_module import (
    AutoEncoderModule,
    is_lightning_checkpoint,
    load_model_state_dict,
    load_module,
    resolve_accelerator,
    verify_checkpoint_schema,
)
from gut_former.utils.log_config import setup_logging

log = logging.getLogger(__name__)


def run_inference(cfg: DictConfig, **_):
    # Setting up project
    setup_logging(logging.INFO if cfg.task.verbose else logging.WARNING)

    if cfg.task.checkpoint_path is None:
        raise ValueError(
            "Inference requires a trained model checkpoint; set checkpoint_path in config."
        )

    # Loading & Preparing Data
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("predict")
    data_schema = datamodule.data_schema()

    # Recreating Model
    log.info("Recreating model from checkpoint: %s\n", cfg.task.checkpoint_path)
    verify_checkpoint_schema(cfg.task.checkpoint_path, data_schema)
    if is_lightning_checkpoint(cfg.task.checkpoint_path):
        module = load_module(cfg.task.checkpoint_path)
    else:
        log.warning(
            "Legacy checkpoint: rebuilding the model from cfg.model and the inference data."
        )
        model = hydra.utils.instantiate(cfg.model)(data_schema=data_schema)
        model.load_state_dict(load_model_state_dict(cfg.task.checkpoint_path))
        module = AutoEncoderModule(model)

    # Inference
    trainer = hydra.utils.instantiate(
        cfg.trainer, accelerator=resolve_accelerator(cfg.trainer.accelerator)
    )
    if not any(isinstance(callback, BasePredictionWriter) for callback in trainer.callbacks):
        log.warning(
            "No prediction writer is configured in trainer.callbacks; nothing will be saved."
        )

    # Outputs are written by the PredictionWriter callback (conf/trainer/inference.yaml)
    trainer.predict(module, datamodule=datamodule, return_predictions=False)
