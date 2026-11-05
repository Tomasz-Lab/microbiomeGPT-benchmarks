import logging
from pathlib import Path

import hydra
import lightning.pytorch as L
from omegaconf import DictConfig, OmegaConf

from gut_former.system import (
    build_model,
    load_model_state_dict,
    verify_checkpoint_schema,
)
from gut_former.utils.log_config import setup_logging

log = logging.getLogger(__name__)


def run_training(cfg: DictConfig, **_: object) -> None:
    # Setting up project
    setup_logging(logging.INFO if cfg.task.verbose else logging.WARNING)

    if cfg.task.resume_from is not None and cfg.task.init_weights is not None:
        raise ValueError("Set either task.resume_from or task.init_weights, not both.")

    L.seed_everything(cfg.task.seed, workers=True)

    # Loading & Preparing Data
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("fit")
    data_schema = datamodule.data_schema()

    for checkpoint_path in (cfg.task.resume_from, cfg.task.init_weights):
        if checkpoint_path is not None:
            verify_checkpoint_schema(checkpoint_path, data_schema)

    # Defining Model
    model = build_model(cfg.model, data_schema)

    if cfg.task.init_weights is not None:
        model.load_state_dict(load_model_state_dict(cfg.task.init_weights))
        log.info("Initialized model weights from: %s", cfg.task.init_weights)

    system = hydra.utils.instantiate(cfg.system)(
        model=model,
        loss_fn=hydra.utils.instantiate(cfg.task.loss),
        optimizer=hydra.utils.instantiate(cfg.task.optimizer),
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        data_schema=data_schema.to_dict(),
    )

    if cfg.task.resume_from is not None:
        # ModelCheckpoint restores its best score only when saving to the same folder, and every
        # Hydra run has a new output folder, so keep writing next to the resumed checkpoint
        checkpoint_dir = str(Path(cfg.task.resume_from).resolve().parent)
        for callback_cfg in cfg.trainer.get("callbacks") or []:
            if callback_cfg.get("_target_", "").endswith(".ModelCheckpoint"):
                callback_cfg.dirpath = checkpoint_dir
        log.info("Resuming: checkpoints continue in %s", checkpoint_dir)

    # Training Model
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(system, datamodule=datamodule, ckpt_path=cfg.task.resume_from)
