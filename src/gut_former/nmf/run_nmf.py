import argparse
import logging

from gut_former import dataset
import pandas as pd

from gut_former.nmf.nmf_core import run_nmf_loop
from gut_former.utils.data_prep import collapse_to_genus, filter_samples, train_val_split
from gut_former.utils.log_config import setup_logging
from gut_former.utils.project_paths import find_data_path, find_output_path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from dotenv import load_dotenv

from pathlib import Path

load_dotenv()

log = logging.getLogger(__name__)


def load_data_nmf_csv(data_path: str, metadata_path: str, kind: str, subset: str, **_) -> pd.DataFrame:
    """Load and preprocess input data based on data type."""
    df = pd.read_csv(data_path, index_col=0, low_memory=False).fillna(0).sort_index()
    metadata = pd.read_csv(metadata_path, index_col=0, low_memory=False).sort_index()

    if kind == "pathways":
        df = df * 100

    df = filter_samples(df, metadata, subset)

    if kind == "taxonomy":
        df = collapse_to_genus(df)

    return df

def load_data_nmf_db():
    pass

@hydra.main(version_base="1.3", config_path="../../../conf", config_name="config")
def main(cfg: DictConfig):
    setup_logging()

    log.info("Starting NMF script..")

    # Load & preprocess
    X = hydra.utils.instantiate(cfg.dataset)
    log.info("Data loaded | shape: %s", X.shape)

    # Train / val split
    X_train, X_val = train_val_split(X)
    log.info("Train: %s | Val: %s", X_train.shape, X_val.shape)

    # Run NMF
    best, history = run_nmf_loop(X_train, X_val, cfg.model.parameters.n_components, cfg.model.parameters.n_runs)

    # Save outputs
    output_path = Path(HydraConfig.get().runtime.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n   = cfg.model.parameters.n_components
    ds  = cfg.dataset.name
    tag = f"{cfg.dataset.kind}_{cfg.dataset.subset}"
    pd.DataFrame(best["H"],     columns=X_train.columns).to_csv(output_path / f"H_{n}_{ds}_{tag}.csv")
    pd.DataFrame(best["W"],     index=X_train.index)    .to_csv(output_path / f"W_train_{n}_{ds}_{tag}.csv")
    pd.DataFrame(best["W_val"], index=X_val.index)      .to_csv(output_path / f"W_val_{n}_{ds}_{tag}.csv")
    pd.DataFrame(history)                           .to_csv(output_path / f"nmf_stats_{n}_{ds}_{tag}.csv", index=False)

    log.info(
        "Done. Saved to %s:\n"
        "  - H_%d_%s_%s.csv\n"
        "  - W_train_%d_%s_%s.csv\n"
        "  - W_val_%d_%s_%s.csv\n"
        "  - nmf_stats_%d_%s_%s.csv",
        output_path, n, ds, tag, n, ds, tag, n, ds, tag, n, ds, tag,
    )


if __name__ == "__main__":
    main()
