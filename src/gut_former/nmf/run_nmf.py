import logging
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from gut_former.nmf.nmf_core import NMF
from gut_former.utils.data_prep import collapse_to_genus, filter_samples, train_val_split
from gut_former.utils.log_config import setup_logging

load_dotenv()

log = logging.getLogger(__name__)


def load_data_nmf_csv(
    data_path: str,
    metadata_path: str,
    kind: str,
    subset: str,
    pathways_multiplier: int,
    **_: object,
) -> pd.DataFrame:
    """Load and preprocess input data based on data type."""
    df = pd.read_csv(data_path, index_col=0, low_memory=False).fillna(0).sort_index()
    metadata = pd.read_csv(metadata_path, index_col=0, low_memory=False).sort_index()

    if kind == "pathways":
        df = df * pathways_multiplier

    df = filter_samples(df, metadata, subset)

    if kind == "taxonomy":
        df = collapse_to_genus(df)

    return df


def load_data_nmf_db() -> None:
    pass


def run_nmf_task(cfg: DictConfig, **_: object) -> None:
    setup_logging()

    log.info("Starting NMF script..")

    # Load & preprocess
    X = hydra.utils.instantiate(cfg.dataset)
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
    ds = cfg.dataset.name
    tag = f"{cfg.dataset.kind}_{cfg.dataset.subset}"
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
