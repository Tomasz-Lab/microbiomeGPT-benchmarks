import re
import sys
from pathlib import Path

import pandas as pd
import sweetviz as sv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def run_data_profiling_report(cfg: DictConfig, **_):
    df = pd.read_csv(cfg.task.path, low_memory=False)
    dropped_columns = []

    logs = []
    orig_stdout_write = sys.stdout.write
    orig_stderr_write = sys.stderr.write
    while True:
        # Stream live to console while capturing both stdout and stderr
        sys.stdout.write = lambda s: (logs.append(s), orig_stdout_write(s))[1]
        sys.stderr.write = lambda s: (logs.append(s), orig_stderr_write(s))[1]

        try:
            report = sv.analyze(df)
            sys.stdout.write = orig_stdout_write
            sys.stderr.write = orig_stderr_write
            break  # if there is no errors - break loop
        except ValueError as e:
            # Restore standard streams immediately so all logs are flushed
            sys.stdout.write = orig_stdout_write
            sys.stderr.write = orig_stderr_write

            full_log = "".join(logs)
            matches = re.findall(r"Feature:\s*([^\s|]+)", full_log)

            if matches and matches[-1] in df.columns:
                col = matches[-1]
                print(f"\n[Sweetviz Fix] Dropping problematic column: {col}\n")
                df = df.drop(columns=[col])
                dropped_columns.append(col)
            else:
                print(f"\n[Sweetviz Error] Couldn't isolate column. Raw error: {e}")
                raise e

    if dropped_columns:
        print(f"Report generated! Dropped {len(dropped_columns)} column(s): {dropped_columns}")

    output_path = Path(HydraConfig.get().runtime.output_dir)
    report.show_html(output_path / f"{cfg.task.output}.html")
