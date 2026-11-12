import logging

import hydra
from omegaconf import DictConfig

from gut_former.system import verify_checkpoint_schema
from gut_former.utils.hydra_utils import load_from_checkpoint
from gut_former.utils.log_config import setup_logging
from gut_former.utils.trainer import ensure_prediction_writer

log = logging.getLogger(__name__)


def run_inference(cfg: DictConfig, **_: object) -> None:
    # Setting up project
    setup_logging(logging.INFO if cfg.task.verbose else logging.WARNING)

    log.info("Starting inference script..")

    if cfg.task.checkpoint_path is None:
        raise ValueError(
            "Inference requires a trained model checkpoint; set checkpoint_path in config."
        )

    # Checked before loading data and the checkpoint, so a bad trainer config fails fast
    trainer = hydra.utils.instantiate(cfg.trainer)
    ensure_prediction_writer(trainer)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("predict")

    verify_checkpoint_schema(cfg.task.checkpoint_path, datamodule.data_schema())

    system = load_from_checkpoint(cfg.system, cfg.task.checkpoint_path)

    trainer.predict(system, datamodule=datamodule, return_predictions=False)
