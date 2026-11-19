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


class NegativeValuesError(ValueError):
    """Raised when some rows of a CompositionalTable contain negative values.

    Attributes:
        modality: Name of the modality that failed validation.
        row_mins: Smallest value of each offending row, indexed by sample id.
    """

    def __init__(self, modality: str, row_mins: pd.Series) -> None:
        self.modality = modality
        self.row_mins = row_mins
        preview = ", ".join(f"{i}={m:.6g}" for i, m in row_mins.head(5).items())
        more = f" (+{len(row_mins) - 5} more)" if len(row_mins) > 5 else ""
        super().__init__(
            f"Modality '{modality}': {len(row_mins)} sample(s) contain negative values "
            f"(all values must be non-negative), min per sample: {preview}{more}"
        )


class NonNumericColumnsError(TypeError):
    """Raised when some columns of a CompositionalTable are not numeric.

    Attributes:
        modality: Name of the modality that failed validation.
        column_dtypes: Dtypes of the offending columns, indexed by column name.
    """

    def __init__(self, modality: str, column_dtypes: pd.Series) -> None:
        self.modality = modality
        self.column_dtypes = column_dtypes
        preview = ", ".join(f"{c}={d}" for c, d in column_dtypes.head(5).items())
        more = f" (+{len(column_dtypes) - 5} more)" if len(column_dtypes) > 5 else ""
        super().__init__(
            f"Modality '{modality}': {len(column_dtypes)} column(s) are not numeric "
            f"(all columns must be numeric): {preview}{more}"
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
        self._check_is_numeric(data, name)
        self._check_non_negative(data, name)
        self._check_rows_sum_to_total(data, name)

        return data

    def _check_is_numeric(self, data: pd.DataFrame, name: str) -> None:
        bad = ~data.dtypes.map(pd.api.types.is_numeric_dtype).astype(bool)
        if bad.any():
            raise NonNumericColumnsError(name, data.dtypes[bad])

    def _check_non_negative(self, data: pd.DataFrame, name: str) -> None:
        bad = data.lt(0).any(axis=1)
        if bad.any():
            raise NegativeValuesError(name, data[bad].min(axis=1))

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
