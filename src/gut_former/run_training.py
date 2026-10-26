import logging

import hydra
import lightning.pytorch as L
from omegaconf import DictConfig, OmegaConf

from gut_former.datamodule import BacteriaDataModule
from gut_former.lit_module import (
    GutFormerModule,
    load_model_state_dict,
    resolve_accelerator,
    verify_checkpoint_features,
)
from gut_former.utils.log_config import setup_logging

log = logging.getLogger(__name__)


def run_training(cfg: DictConfig, **_):
    # Setting up project
    setup_logging(logging.INFO if cfg.task.verbose else logging.WARNING)

    log.info("Starting training script..\n ")

    if cfg.task.resume_from is not None and cfg.task.init_weights is not None:
        raise ValueError("Set either task.resume_from or task.init_weights, not both.")

    L.seed_everything(cfg.task.seed, workers=True)

    # Loading & Preparing Data
    datamodule = BacteriaDataModule(
        name=cfg.dataset.name,
        taxa_multiplier=cfg.dataset.taxa_multiplier,
        pathways_multiplier=cfg.dataset.pathways_multiplier,
        batch_size=cfg.task.batch_size,
    )
    datamodule.setup("fit")

    for checkpoint_path in (cfg.task.resume_from, cfg.task.init_weights):
        if checkpoint_path is not None:
            verify_checkpoint_features(checkpoint_path, datamodule.feature_names)

    # Defining Model
    model_kwargs = {
        "num_pathways": datamodule.num_pathways,
        "num_bacteria": datamodule.num_bacteria,
    }
    model = hydra.utils.instantiate(cfg.model)(**model_kwargs)

    if cfg.task.init_weights is not None:
        model.load_state_dict(load_model_state_dict(cfg.task.init_weights))
        log.info("Initialized model weights from: %s", cfg.task.init_weights)

    module = GutFormerModule(
        model=model,
        loss_fn=hydra.utils.instantiate(cfg.task.loss),
        optimizer=hydra.utils.instantiate(cfg.task.optimizer),
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        model_kwargs=model_kwargs,
        feature_names=datamodule.feature_names,
    )

    # Training Model
    trainer = hydra.utils.instantiate(
        cfg.trainer, accelerator=resolve_accelerator(cfg.trainer.accelerator)
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.task.resume_from)

    checkpoint_callback = trainer.checkpoint_callback
    log.info(
        "\nTraining completed.\nSaved checkpoints:\n  - Best: %s\n  - Last: %s\n",
        checkpoint_callback.best_model_path if checkpoint_callback else None,
        checkpoint_callback.last_model_path if checkpoint_callback else None,
    )
