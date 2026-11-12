from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from data.modalities.base import ModalityKind

type MissingPolicy = Literal["fill_zero", "error", "keep_nan"]


@dataclass(frozen=True)
class CompositionalTable(ModalityKind):
    """Table with one row per sample whose non-negative values sum to the same total.

    Attributes:
        total: Value every row sums to (e.g. 1.0 for relative abundance, 100 for percent).
        tolerance: Allowed absolute deviation of a row sum from `total`.
        missing: How missing values are handled:
            "fill_zero" replaces them with 0 (missing means absent),
            "error" rejects data that has any,
            "keep_nan" keeps them; rows with NaN are excluded from the row-sum check.
    """

    total: float
    tolerance: float
    feature_pattern: str | None = None
    missing: MissingPolicy = "error"

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError(f"total must be positive, got {self.total}")
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}")

    def validate(self, data: pd.DataFrame, name: str) -> pd.DataFrame:
        pass


@dataclass(frozen=True)
class Graph(ModalityKind):
    directed: bool

    def validate(self, data: Any, name: str) -> Any:
        raise NotImplementedError("Graph modalities are not supported yet.")
