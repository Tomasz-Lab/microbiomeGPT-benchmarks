import logging
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from data.load import load_dataset
from data.modataset.base import MultiOmicsDataset
from mgptbench.nmf.nmf_core import NMF
from mgptbench.utils.data_prep import collapse_to_genus as to_genus
from mgptbench.utils.data_prep import filter_samples, train_val_split
from mgptbench.utils.log_config import setup_logging

load_dotenv()

log = logging.getLogger(__name__)


def prepare_nmf_input(
    dataset: MultiOmicsDataset,
    modality: str,
    subset: str,
    multiplier: float,
    collapse_to_genus: bool,
) -> pd.DataFrame:
    """Select one modality of the dataset and turn it into the NMF input matrix."""
    df = dataset[modality].data.sort_index() * multiplier
    df = filter_samples(df, dataset.metadata, subset)
    if collapse_to_genus:
        df = to_genus(df)
    return df


def load_nmf_input(cfg: DictConfig) -> pd.DataFrame:
    """Load the configured dataset and prepare the NMF input from `cfg.task`."""
    dataset = load_dataset(hydra.utils.instantiate(cfg.modataset))
    return prepare_nmf_input(
        dataset,
        modality=cfg.task.modality,
        subset=cfg.task.subset,
        multiplier=cfg.task.multiplier,
        collapse_to_genus=cfg.task.collapse_to_genus,
    )


def run_nmf_task(cfg: DictConfig, **_: object) -> None:
    setup_logging()

    log.info("Starting NMF script..")

    # Load & preprocess
    X = load_nmf_input(cfg)
    log.info("Data loaded | shape: %s", X.shape)

    # Train / val split
    X_train, X_val = train_val_split(X)
    log.info("Train: %s | Val: %s", X_train.shape, X_val.shape)

    # Run NMF
    nmf_model = NMF(**cfg.model.parameters).fit(X_train)
    nmf_eval_results = nmf_model.evaluate(X_val)

    # Save outputs
    output_path = Path(HydraConfig.get().runtime.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n = cfg.model.parameters.n_components
    ds = cfg.modataset.name
    tag = f"{cfg.task.modality}_{cfg.task.subset}"
    pd.DataFrame(nmf_model.H_, columns=X_train.columns).to_csv(
        output_path / f"H_{n}_{ds}_{tag}.csv"
    )
    pd.DataFrame(nmf_model.W_train_, index=X_train.index).to_csv(
        output_path / f"W_train_{n}_{ds}_{tag}.csv"
    )
    pd.DataFrame(nmf_eval_results["W_val"], index=X_val.index).to_csv(
        output_path / f"W_val_{n}_{ds}_{tag}.csv"
    )
    pd.DataFrame(nmf_model.history_).to_csv(
        output_path / f"nmf_stats_{n}_{ds}_{tag}.csv", index=False
    )

    log.info(
        "Done. Saved to %s:\n"
        "  - H_%d_%s_%s.csv\n"
        "  - W_train_%d_%s_%s.csv\n"
        "  - W_val_%d_%s_%s.csv\n"
        "  - nmf_stats_%d_%s_%s.csv",
        output_path,
        n,
        ds,
        tag,
        n,
        ds,
        tag,
        n,
        ds,
        tag,
        n,
        ds,
        tag,
    )
