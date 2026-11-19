import pytest

from data.modalities.kinds import CompositionalTable, RowsNotSumToTotalError

KIND = CompositionalTable(total=1.0, tolerance=1e-6, missing="error")


def test_check_rows_sum_to_total_accepts_valid_data(compositional_table):
    KIND._check_rows_sum_to_total(compositional_table, "taxonomy")


def test_check_rows_sum_to_total_accepts_row_sums_within_tolerance(
    compositional_table_within_tolerance,
):
    kind = CompositionalTable(total=1.0, tolerance=1e-2, missing="error")

    kind._check_rows_sum_to_total(compositional_table_within_tolerance, "taxonomy")


def test_check_rows_sum_to_total_rejects_rows_not_summing_to_total(compositional_table_wrong_sum):
    with pytest.raises(RowsNotSumToTotalError) as exc_info:
        KIND._check_rows_sum_to_total(compositional_table_wrong_sum, "taxonomy")

    assert list(exc_info.value.row_sums.index) == ["s2", "s3"]


def test_check_non_negative_accepts_non_negative_data(compositional_table):
    KIND._check_non_negative(compositional_table, "taxonomy")


def test_check_non_negative_rejects_negative_values(compositional_table_negative):
    with pytest.raises(ValueError, match="non-negative"):
        KIND._check_non_negative(compositional_table_negative, "taxonomy")
