import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from data.modalities.base import ModalityKind


class ModalityValidationError(ValueError):
    """Base for CompositionalTable validation errors.

    Attributes:
        modality: Name of the modality that failed validation.
        offenders: Offending samples (or columns) with the value that explains why.
    """

    def __init__(self, modality: str, offenders: pd.Series, problem: str, fmt: str = "") -> None:
        self.modality = modality
        self.offenders = offenders
        preview = ", ".join(f"{i}={v:{fmt}}" for i, v in offenders.head(5).items())
        more = f" (+{len(offenders) - 5} more)" if len(offenders) > 5 else ""
        super().__init__(f"Modality '{modality}': {len(offenders)} {problem}: {preview}{more}")


class RowsNotSumToTotalError(ModalityValidationError):
    """Raised when some rows do not sum to `total`; offenders are the row sums, by sample id."""

    def __init__(self, modality: str, row_sums: pd.Series, total: float, tolerance: float) -> None:
        self.total = total
        self.tolerance = tolerance
        super().__init__(
            modality, row_sums, f"sample(s) do not sum to {total} ± {tolerance}", ".6g"
        )


class NegativeValuesError(ModalityValidationError):
    """Raised when some rows contain negative values; offenders are the row minima, by sample id."""

    def __init__(self, modality: str, row_mins: pd.Series) -> None:
        super().__init__(
            modality, row_mins, "sample(s) contain negative values, min per sample", ".6g"
        )


class NonNumericColumnsError(ModalityValidationError, TypeError):
    """Raised when some columns are not int or float; offenders are their dtypes, by column."""

    def __init__(self, modality: str, column_dtypes: pd.Series) -> None:
        super().__init__(modality, column_dtypes, "column(s) are not int or float")


class MissingValuesError(ModalityValidationError):
    """Raised when some rows contain NaN; offenders are the NaN counts, by sample id."""

    def __init__(self, modality: str, missing_counts: pd.Series) -> None:
        super().__init__(
            modality, missing_counts, "sample(s) contain missing values (missing='error')"
        )


class InvalidColumnNamesError(ModalityValidationError):
    """Raised when some column names do not match `column_pattern`; offenders are their positions, by name."""

    def __init__(self, modality: str, positions: pd.Series, pattern: str) -> None:
        self.pattern = pattern
        super().__init__(
            modality, positions, f"column name(s) do not match pattern {pattern!r}, position"
        )


@dataclass(frozen=True)
class CompositionalTable(ModalityKind):
    """Table with one row per sample whose non-negative values sum to the same total.

    Attributes:
        total: Value every row sums to (e.g. 1.0 for relative abundance, 100 for percent).
        tolerance: Allowed absolute deviation of a row sum from `total`.
        column_pattern: Regex every column name must fully match (e.g. MetaPhlAn 'k__...|s__...').
            Catches tables from another tool, mixed naming conventions and empty or unnamed columns.
            None skips the check.
    """

    total: float
    tolerance: float
    column_pattern: str | None = None

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError(f"total must be positive, got {self.total}")
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}")
        if self.column_pattern is not None:
            re.compile(self.column_pattern)

    def validate(self, data: pd.DataFrame, name: str) -> pd.DataFrame:
        if self.column_pattern is not None:
            self._check_column_names(data, name)
        self._check_is_numeric(data, name)
        self._check_no_missing(data, name)
        self._check_non_negative(data, name)
        self._check_rows_sum_to_total(data, name)

        return data

    def _check_column_names(self, data: pd.DataFrame, name: str) -> None:
        pattern = re.compile(self.column_pattern)
        bad = [
            (column, position)
            for position, column in enumerate(data.columns)
            if not isinstance(column, str) or pattern.fullmatch(column) is None
        ]
        if bad:
            positions = pd.Series([pos for _, pos in bad], index=[repr(col) for col, _ in bad])
            raise InvalidColumnNamesError(name, positions, self.column_pattern)

    def _check_is_numeric(self, data: pd.DataFrame, name: str) -> None:
        # Only int or float columns are allowed; bool, complex, strings etc. are rejected.
        is_int_or_float = data.dtypes.map(
            lambda d: pd.api.types.is_integer_dtype(d) or pd.api.types.is_float_dtype(d)
        )
        bad = ~is_int_or_float.astype(bool)
        if bad.any():
            raise NonNumericColumnsError(name, data.dtypes[bad])

    def _check_no_missing(self, data: pd.DataFrame, name: str) -> None:
        missing_counts = data.isna().sum(axis=1)
        bad = missing_counts > 0
        if bad.any():
            raise MissingValuesError(name, missing_counts[bad])

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
