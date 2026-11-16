from typing import Protocol

import pandas as pd


class Source(Protocol):
    """Where the data lives and how to read it (CSV, Parquet, database, ...).

    Returns a raw table with no domain meaning; knows nothing about modalities.
    """

    def read(self) -> pd.DataFrame: ...
