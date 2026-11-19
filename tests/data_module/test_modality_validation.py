import pandas as pd
import pytest

from data.modalities.kinds import CompositionalTable, RowsNotSumToTotalError

KIND = CompositionalTable(total=1.0, tolerance=1e-6, missing="error")


def test_compositional_table_accepts_valid_data(compositional_table):
    result = KIND.validate(compositional_table, "taxonomy")

    pd.testing.assert_frame_equal(result, compositional_table)


def test_compositional_table_accepts_row_sums_within_tolerance(
    compositional_table_within_tolerance,
):
    kind = CompositionalTable(total=1.0, tolerance=1e-2, missing="error")

    result = kind.validate(compositional_table_within_tolerance, "taxonomy")

    pd.testing.assert_frame_equal(result, compositional_table_within_tolerance)


def test_compositional_table_rejects_rows_not_summing_to_total(compositional_table_wrong_sum):
    with pytest.raises(RowsNotSumToTotalError) as exc_info:
        KIND.validate(compositional_table_wrong_sum, "taxonomy")

    assert list(exc_info.value.row_sums.index) == ["s2", "s3"]
