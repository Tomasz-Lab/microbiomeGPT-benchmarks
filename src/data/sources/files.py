from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.sources.base import Source, SourceInfo


@dataclass(frozen=True)
class CsvSource(Source[pd.DataFrame]):
    """Reads a CSV file into a raw DataFrame."""

    path: str | Path

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def describe(self) -> SourceInfo:
        return SourceInfo(location=str(Path(self.path).resolve()))


@dataclass(frozen=True)
class ParquetSource(Source[pd.DataFrame]):
    """Reads a Parquet file into a raw DataFrame."""

    path: str | Path

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

    def describe(self) -> SourceInfo:
        return SourceInfo(location=str(Path(self.path).resolve()))
