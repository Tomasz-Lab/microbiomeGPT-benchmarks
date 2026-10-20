import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from gut_former.utils.log_config import setup_logging
from gut_former.utils.project_paths import find_data_path

log = logging.getLogger(__name__)


def run_inference(cfg: DictConfig, **_):
    # Setting up project
    setup_logging()

    log.info("Starting inference script..\n ")

    if cfg.task.checkpoint_path is None:
        raise ValueError("Inference requires a trained model checkpoint; set task.checkpoint_path.")

    output_path = Path(HydraConfig.get().runtime.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Loading & Preparing Data
    data_path = find_data_path()
    t_df = (
        pd.read_csv(f"{data_path}/taxonomy_{cfg.dataset.name}.csv", index_col=[0], low_memory=False)
        .fillna(0)
        .sort_index()
        * cfg.dataset.taxa_multiplier
    )
    p_df = (
        pd.read_csv(f"{data_path}/pathways_{cfg.dataset.name}.csv", index_col=[0], low_memory=False)
        .fillna(0)
        .sort_index()
        * cfg.dataset.pathways_multiplier
    )

    t_tensor = torch.tensor(t_df.values, dtype=torch.float32).unsqueeze(-1)
    p_tensor = torch.tensor(p_df.values, dtype=torch.float32).unsqueeze(-1)

    # Recreating Model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_builder = hydra.utils.instantiate(
        cfg.model,
    )

    model = model_builder(
        num_pathways=p_df.shape[1],
        num_bacteria=t_df.shape[1],
    ).to(DEVICE)

    log.info("Recreating model from checkpoint: %s\n", cfg.task.checkpoint_path)
    checkpoint_dict = torch.load(cfg.task.checkpoint_path, weights_only=True, map_location=DEVICE)
    model.load_state_dict(checkpoint_dict)
    model.eval()
    model.cpu()

    # Inference
    t_pred, p_pred, latent = model(p_tensor, t_tensor)

    # Storing outputs
    t_pred_df = pd.DataFrame(
        t_pred.detach().cpu().numpy().squeeze(-1), index=t_df.index, columns=t_df.columns
    )
    t_pred_df.index.name = t_df.index.name
    t_output_path = f"{output_path}/pred_taxonomy_{cfg.dataset.name}.csv"
    t_pred_df.to_csv(t_output_path, index=True)

    p_pred_df = pd.DataFrame(
        p_pred.detach().cpu().numpy().squeeze(-1), index=p_df.index, columns=p_df.columns
    )
    p_pred_df.index.name = p_df.index.name
    p_output_path = f"{output_path}/pred_pathways_{cfg.dataset.name}.csv"
    p_pred_df.to_csv(p_output_path, index=True)

    latent_cols = [f"z{i}" for i in range(cfg.model.latent_size)]
    latent_df = pd.DataFrame(latent.detach().cpu().numpy(), index=p_df.index, columns=latent_cols)
    latent_df.index.name = t_df.index.name
    latent_output_path = f"{output_path}/latent_{cfg.dataset.name}.csv"
    latent_df.to_csv(latent_output_path, index=True)

    log.info(
        "Inference completed.\nSaved outputs:\n  - Latent: %s\n  - P_pred:  %s\n  - T_pred:  %s",
        latent_output_path,
        p_output_path,
        t_output_path,
    )
