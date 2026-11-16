from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from data.modalities.base import ModalityKind, ModalityValidationError

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
    missing: MissingPolicy = "error"

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError(f"total must be positive, got {self.total}")
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}")

    def validate(self, data: pd.DataFrame, name: str) -> pd.DataFrame:
        # Minimal version: first column is the sample id, the rest are non-negative values with a fixed row sum.
        table = data.set_index(data.columns[0]).rename_axis("sample_id")
        if self.missing == "fill_zero":
            table = table.fillna(0)
        elif self.missing == "error" and table.isna().any().any():
            raise ModalityValidationError(f"{name}: has missing values")
        if (table < 0).any().any():
            raise ModalityValidationError(f"{name}: has negative values")
        off = ((table.sum(axis=1) - self.total).abs() > self.tolerance) & table.notna().all(axis=1)
        if off.any():
            raise ModalityValidationError(
                f"{name}: rows must sum to {self.total}, got {list(off[off].index)}"
            )
        return table

    def sample_ids(self, data: pd.DataFrame) -> pd.Index:
        return data.index

    def select_samples(self, data: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
        pass


@dataclass(frozen=True)
class Graph(ModalityKind):
    directed: bool

    def validate(self, data: Any, name: str) -> Any:
        raise NotImplementedError("Graph modalities are not supported yet.")

    def sample_ids(self, data: Any) -> pd.Index:
        pass

    def select_samples(self, data: Any, ids: Sequence[str]) -> Any:
        pass
