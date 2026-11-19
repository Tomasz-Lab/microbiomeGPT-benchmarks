from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from data.modalities.base import ModalityKind

type MissingPolicy = Literal["fill_zero", "error", "keep_nan"]


class RowsNotSumToTotalError(ValueError):
    """Raised when some rows of a CompositionalTable do not sum to its total.

    Attributes:
        modality: Name of the modality that failed validation.
        row_sums: Sums of the offending rows, indexed by sample id.
        total: Expected row sum.
        tolerance: Allowed absolute deviation from `total`.
    """

    def __init__(self, modality: str, row_sums: pd.Series, total: float, tolerance: float) -> None:
        self.modality = modality
        self.row_sums = row_sums
        self.total = total
        self.tolerance = tolerance
        preview = ", ".join(f"{i}={s:.6g}" for i, s in row_sums.head(5).items())
        more = f" (+{len(row_sums) - 5} more)" if len(row_sums) > 5 else ""
        super().__init__(
            f"Modality '{modality}': {len(row_sums)} sample(s) do not sum to "
            f"{total} ± {tolerance}: {preview}{more}"
        )


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
        self._check_non_negative(data, name)
        self._check_rows_sum_to_total(data, name)

        return data

    def _check_non_negative(self, data: pd.DataFrame, name: str) -> None:
        if not (data.to_numpy() >= 0).all():
            raise ValueError(f"Modality '{name}': all values must be non-negative")

    def _check_rows_sum_to_total(self, data: pd.DataFrame, name: str) -> None:
        # skipna=False: a row with NaN sums to NaN, and NaN > x is False,
        # so such rows are excluded from the check (consistent with "keep_nan").
        row_sums = data.sum(axis=1, skipna=False)
        bad = (row_sums - self.total).abs() > self.tolerance
        if bad.any():
            raise RowsNotSumToTotalError(name, row_sums[bad], self.total, self.tolerance)

    def select_samples(self, data: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
        return data.loc[data.index.intersection(ids)]


@dataclass(frozen=True)
class Graph(ModalityKind):
    directed: bool

    def validate(self, data: Any, name: str) -> Any:
        raise NotImplementedError("Graph modalities are not supported yet.")

    def sample_ids(self, data: Any) -> pd.Index:
        pass

    def select_samples(self, data: Any, ids: Sequence[str]) -> Any:
        pass
