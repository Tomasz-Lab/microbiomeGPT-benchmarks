import logging

import hydra
import lightning.pytorch as L
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from gut_former.system import verify_checkpoint_schema
from gut_former.utils.environment import environment_state
from gut_former.utils.git_state import git_state
from gut_former.utils.hydra_utils import cfg_to_plain_dict
from gut_former.utils.log_config import setup_logging

log = logging.getLogger(__name__)


def run_training(cfg: DictConfig, **_: object) -> None:
    # Setting up project
    setup_logging(logging.INFO if cfg.task.verbose else logging.WARNING)

    log.info("Starting training script..")

    if cfg.task.resume_from is not None and cfg.task.init_weights is not None:
        raise ValueError("Set either task.resume_from or task.init_weights, not both.")

    L.seed_everything(cfg.task.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("fit")
    data_schema = datamodule.data_schema()

    if cfg.task.resume_from is not None:
        verify_checkpoint_schema(cfg.task.resume_from, data_schema)

    model_schema = cfg_to_plain_dict(cfg.model)
    model_schema["data_schema"] = data_schema.to_config()

    # D: 0001
    system = hydra.utils.instantiate(cfg.system)(
        model_schema=model_schema,
    )

    trainer = hydra.utils.instantiate(cfg.trainer)

    # D: 0003
    if trainer.loggers:
        hparams = {
            "run": cfg_to_plain_dict(cfg.run),
            "code": git_state(HydraConfig.get().runtime.output_dir),
            "env": environment_state(trainer),
        }
        for logger in trainer.loggers:
            logger.log_hyperparams(hparams)

    trainer.fit(system, datamodule=datamodule, ckpt_path=cfg.task.resume_from)
