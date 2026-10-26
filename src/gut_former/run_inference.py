import logging
from pathlib import Path

import hydra
import lightning.pytorch as L
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from gut_former.datamodule import BacteriaDataModule
from gut_former.lit_module import (
    GutFormerModule,
    is_lightning_checkpoint,
    load_model_state_dict,
    load_module,
    resolve_accelerator,
    verify_checkpoint_features,
)
from gut_former.utils.log_config import setup_logging

log = logging.getLogger(__name__)


def run_inference(cfg: DictConfig, **_):
    # Setting up project
    setup_logging(logging.INFO if cfg.task.verbose else logging.WARNING)

    log.info("Starting inference script..\n ")

    if cfg.task.checkpoint_path is None:
        raise ValueError(
            "Inference requires a trained model checkpoint; set checkpoint_path in config."
        )

    output_path = Path(HydraConfig.get().runtime.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Loading & Preparing Data
    datamodule = BacteriaDataModule(
        name=cfg.dataset.name,
        taxa_multiplier=cfg.dataset.taxa_multiplier,
        pathways_multiplier=cfg.dataset.pathways_multiplier,
        batch_size=cfg.task.batch_size,
    )
    datamodule.setup("predict")

    # Recreating Model
    log.info("Recreating model from checkpoint: %s\n", cfg.task.checkpoint_path)
    verify_checkpoint_features(cfg.task.checkpoint_path, datamodule.feature_names)
    if is_lightning_checkpoint(cfg.task.checkpoint_path):
        module = load_module(cfg.task.checkpoint_path)
    else:
        log.warning(
            "Legacy checkpoint: rebuilding the model from cfg.model and the inference data."
        )
        model = hydra.utils.instantiate(cfg.model)(
            num_pathways=datamodule.num_pathways,
            num_bacteria=datamodule.num_bacteria,
        )
        model.load_state_dict(load_model_state_dict(cfg.task.checkpoint_path))
        module = GutFormerModule(model)

    # Inference
    trainer = L.Trainer(
        accelerator=resolve_accelerator(cfg.task.accelerator),
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    batches = trainer.predict(module, datamodule=datamodule)
    if not batches:
        raise RuntimeError("Prediction returned no batches; is the inference dataset empty?")
    preds = {key: torch.cat([batch[key] for batch in batches]).cpu().numpy() for key in batches[0]}

    # Storing outputs
    t_df, p_df = datamodule.taxonomy, datamodule.pathways

    t_pred_df = pd.DataFrame(preds["taxonomy"], index=t_df.index, columns=t_df.columns)
    t_pred_df.index.name = t_df.index.name
    t_output_path = f"{output_path}/pred_taxonomy_{cfg.dataset.name}.csv"
    t_pred_df.to_csv(t_output_path, index=True)

    p_pred_df = pd.DataFrame(preds["pathways"], index=p_df.index, columns=p_df.columns)
    p_pred_df.index.name = p_df.index.name
    p_output_path = f"{output_path}/pred_pathways_{cfg.dataset.name}.csv"
    p_pred_df.to_csv(p_output_path, index=True)

    latent_cols = [f"z{i}" for i in range(preds["latent"].shape[1])]
    latent_df = pd.DataFrame(preds["latent"], index=p_df.index, columns=latent_cols)
    latent_df.index.name = t_df.index.name
    latent_output_path = f"{output_path}/latent_{cfg.dataset.name}.csv"
    latent_df.to_csv(latent_output_path, index=True)

    log.info(
        "Inference completed.\nSaved outputs:\n  - Latent: %s\n  - P_pred:  %s\n  - T_pred:  %s",
        latent_output_path,
        p_output_path,
        t_output_path,
    )
