import logging
from datetime import date
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from gut_former.dataset import BacteriaDataset
from gut_former.utils.data_prep import train_val_split
from gut_former.utils.log_config import setup_logging
from gut_former.utils.project_paths import find_data_path

log = logging.getLogger(__name__)


def run_training(cfg: DictConfig, **_):
    # Setting up project
    setup_logging(logging.INFO if cfg.task.verbose else logging.WARNING)

    log.info("Starting training script..\n ")

    today_str = date.today().strftime("%Y%m%d")
    output_path = Path(HydraConfig.get().runtime.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats_path = f"{output_path}/{today_str}_{cfg.dataset.name}_training_stats.csv"

    # Loading & Preparing Data
    # To do: refactor later when we will be dealing with task for dataload refacotr
    data_path = find_data_path()
    Xt_df = (
        pd.read_csv(f"{data_path}/taxonomy_{cfg.dataset.name}.csv", index_col=[0], low_memory=False)
        .fillna(0)
        .sort_index()
        * cfg.dataset.taxa_multiplier
    )
    Xp_df = (
        pd.read_csv(f"{data_path}/pathways_{cfg.dataset.name}.csv", index_col=[0], low_memory=False)
        .fillna(0)
        .sort_index()
        * cfg.dataset.pathways_multiplier
    )

    Xt_train, Xt_test = train_val_split(Xt_df)
    Xp_train, Xp_test = train_val_split(Xp_df)

    train_dataset = BacteriaDataset(Xt_train, Xp_train)
    train_dloader = DataLoader(train_dataset, batch_size=cfg.task.batch_size, shuffle=True)

    test_dataset = BacteriaDataset(Xt_test, Xp_test)
    test_dloader = DataLoader(test_dataset, batch_size=cfg.task.batch_size, shuffle=False)

    # Defining Model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_builder = hydra.utils.instantiate(
        cfg.model,
    )

    model = model_builder(
        num_pathways=Xp_df.shape[1],
        num_bacteria=Xt_df.shape[1],
    ).to(DEVICE)

    optimizer_fn = hydra.utils.instantiate(cfg.task.optimizer)
    optimizer = optimizer_fn(model.parameters())

    loss_func = hydra.utils.instantiate(cfg.task.loss)

    if cfg.task.checkpoint_path != "none":
        model.load_state_dict(torch.load(cfg.task.checkpoint_path, map_location=DEVICE))
        log.info("Resumed from checkpoint: %s", cfg.task.checkpoint_path)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info(
        "Model summary:\n  - Total params:      %d\n  - Trainable params:  %d\n",
        total_params,
        trainable_params,
    )

    # Training Model
    history = []
    for epoch in range(cfg.task.epochs):
        model.train()

        train_loss = 0.0
        train_n_samples = 0
        train_t_mse = 0.0
        train_p_mse = 0.0
        for Xt_b, Xp_b in train_dloader:
            batch_size = Xt_b.size(0)
            train_n_samples += batch_size

            Xt_b = Xt_b.to(DEVICE)
            Xp_b = Xp_b.to(DEVICE)

            optimizer.zero_grad()

            Xt_pred, Xp_pred, latent = model(Xp_b, Xt_b)

            loss = loss_func(Xt_b, Xt_pred, Xp_b, Xp_pred)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            train_t_mse += mse_loss(Xt_pred, Xt_b).item() * batch_size
            train_p_mse += mse_loss(Xp_pred, Xp_b).item() * batch_size

        model.eval()

        with torch.no_grad():
            test_loss = 0.0
            test_n_samples = 0
            test_t_mse = 0.0
            test_p_mse = 0.0
            for Xt_b, Xp_b in test_dloader:
                batch_size = Xt_b.size(0)
                test_n_samples += batch_size

                Xt_b = Xt_b.to(DEVICE)
                Xp_b = Xp_b.to(DEVICE)

                Xt_pred, Xp_pred, latent = model(Xp_b, Xt_b)
                loss = loss_func(Xt_b, Xt_pred, Xp_b, Xp_pred)

                test_loss += loss.item() * batch_size
                test_t_mse += mse_loss(Xt_pred, Xt_b).item() * batch_size
                test_p_mse += mse_loss(Xp_pred, Xp_b).item() * batch_size

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss / train_n_samples,
            "train_taxonomy_mse": train_t_mse / train_n_samples,
            "train_pathways_mse": train_p_mse / train_n_samples,
            "test_loss": test_loss / test_n_samples,
            "test_taxonomy_mse": test_t_mse / test_n_samples,
            "test_pathways_mse": test_p_mse / test_n_samples,  # <- note: use test_p_mse here
        }
        history.append(epoch_metrics)

        if epoch % 10 == 1:
            log.info(f"Epoch: {epoch} -> Test Loss: {test_loss:.3f}")
            torch.save(model.state_dict(), cfg.task.checkpoint_path)

    stats_cols = [
        "epoch",
        "train_loss",
        "train_taxonomy_mse",
        "train_pathways_mse",
        "test_loss",
        "test_taxonomy_mse",
        "test_pathways_mse",
    ]

    stats_df = pd.DataFrame(history, columns=stats_cols)
    stats_df.to_csv(stats_path, index=False)

    torch.save(model.state_dict(), cfg.task.checkpoint_path)
    log.info(
        "\nTraining completed.\nSaved outputs:\n  - Training stats: %s\n  - Model checkpoint: %s\n",
        stats_path,
        cfg.task.checkpoint_path,
    )
