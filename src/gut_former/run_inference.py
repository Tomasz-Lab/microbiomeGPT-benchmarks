import logging

import hydra
from omegaconf import DictConfig

from gut_former.system import verify_checkpoint_schema
from gut_former.utils.hydra_utils import load_from_checkpoint
from gut_former.utils.log_config import setup_logging

log = logging.getLogger(__name__)


def run_inference(cfg: DictConfig, **_):
    setup_logging(logging.INFO if cfg.task.verbose else logging.WARNING)

    if cfg.task.checkpoint_path is None:
        raise ValueError(
            "Inference requires a trained model checkpoint; set checkpoint_path in config."
        )

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("predict")

    verify_checkpoint_schema(cfg.task.checkpoint_path, datamodule.data_schema())

    system = load_from_checkpoint(cfg.system, cfg.task.checkpoint_path)

    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.predict(system, datamodule=datamodule, return_predictions=False)
