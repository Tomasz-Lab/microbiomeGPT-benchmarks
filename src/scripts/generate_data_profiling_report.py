import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import TextIOBase
from pathlib import Path
from typing import TextIO

import pandas as pd
import sweetviz as sv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


class _TeeTextIO(TextIOBase):
    def __init__(self, stream: TextIO, logs: list[str]) -> None:
        self._stream = stream
        self._logs = logs

    def write(self, text: str) -> int:
        self._logs.append(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()


def run_data_profiling_report(cfg: DictConfig, **_: object) -> None:
    df = pd.read_csv(cfg.task.path, low_memory=False)
    dropped_columns = []

    logs: list[str] = []
    while True:
        try:
            with (
                redirect_stdout(_TeeTextIO(sys.stdout, logs)),
                redirect_stderr(_TeeTextIO(sys.stderr, logs)),
            ):
                report = sv.analyze(df)
            break  # if there is no errors - break loop
        except ValueError as e:
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
