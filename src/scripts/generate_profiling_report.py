from pathlib import Path

import pandas as pd
import sweetviz as sv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def run_data_profiling_report(cfg: DictConfig, **_):
    df = pd.read_csv(cfg.task.path)
    df = df.drop(columns="uncurated_metadata")
    report = sv.analyze(df)
    output_path = Path(HydraConfig.get().runtime.output_dir)
    report.show_html(output_path / f"{cfg.task.output}.html")
