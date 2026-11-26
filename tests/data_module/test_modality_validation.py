import pytest

from data.modalities.kinds import (
    CompositionalTable,
    DuplicateSampleIdsError,
    InvalidColumnNamesError,
    MissingValuesError,
    NegativeValuesError,
    NonNumericColumnsError,
    RowsNotSumToTotalError,
)

KIND = CompositionalTable(total=1.0)


def test_check_rows_sum_to_total_accepts_valid_data(compositional_table):
    KIND._check_rows_sum_to_total(compositional_table, "taxonomy")


def test_check_rows_sum_to_total_accepts_row_sums_within_tolerance(
    compositional_table_within_tolerance,
):
    kind = CompositionalTable(total=1.0, rtol=1e-2)

    kind._check_rows_sum_to_total(compositional_table_within_tolerance, "taxonomy")


def test_check_rows_sum_to_total_rejects_rows_not_summing_to_total(compositional_table_wrong_sum):
    with pytest.raises(RowsNotSumToTotalError) as exc_info:
        KIND._check_rows_sum_to_total(compositional_table_wrong_sum, "taxonomy")

    assert list(exc_info.value.offenders.index) == ["s2", "s3"]


def test_check_non_negative_accepts_non_negative_data(compositional_table):
    KIND._check_non_negative(compositional_table, "taxonomy")


def test_check_non_negative_rejects_negative_values(compositional_table_negative):
    with pytest.raises(NegativeValuesError) as exc_info:
        KIND._check_non_negative(compositional_table_negative, "taxonomy")

    assert list(exc_info.value.offenders.index) == ["s2", "s3"]


def test_check_is_numeric_accepts_numeric_data(compositional_table):
    KIND._check_is_numeric(compositional_table, "taxonomy")


def test_check_is_numeric_rejects_strings_mixed_with_ints(compositional_table_mixed_types):
    with pytest.raises(NonNumericColumnsError) as exc_info:
        KIND._check_is_numeric(compositional_table_mixed_types, "taxonomy")

    assert list(exc_info.value.offenders.index) == ["species_b"]


def test_check_is_numeric_rejects_bool(compositional_table_bool):
    with pytest.raises(NonNumericColumnsError) as exc_info:
        KIND._check_is_numeric(compositional_table_bool, "taxonomy")

    assert list(exc_info.value.offenders.index) == ["species_b"]


def test_check_no_missing_accepts_data_without_nan(compositional_table):
    KIND._check_no_missing(compositional_table, "taxonomy")


def test_check_no_missing_rejects_nan(compositional_table_missing):
    with pytest.raises(MissingValuesError) as exc_info:
        KIND._check_no_missing(compositional_table_missing, "taxonomy")

    assert list(exc_info.value.offenders.index) == ["s2", "s3"]


def test_check_column_names_accepts_matching_names(compositional_table):
    kind = CompositionalTable(total=1.0, column_pattern=r"species_\w+")

    kind._check_column_names(compositional_table, "taxonomy")


def test_check_column_names_rejects_names_not_matching_pattern(
    compositional_table_bad_column_names,
):
    kind = CompositionalTable(total=1.0, column_pattern=r"species_\w+")

    with pytest.raises(InvalidColumnNamesError) as exc_info:
        kind._check_column_names(compositional_table_bad_column_names, "taxonomy")

    assert list(exc_info.value.offenders.index) == ["'k__Bacteria|s__Prevotella_copri'", "''"]


def test_check_unique_sample_ids_accepts_unique_ids(compositional_table):
    KIND._check_unique_sample_ids(compositional_table, "taxonomy")


def test_check_unique_sample_ids_rejects_duplicates(compositional_table_duplicate_ids):
    with pytest.raises(DuplicateSampleIdsError) as exc_info:
        KIND._check_unique_sample_ids(compositional_table_duplicate_ids, "taxonomy")

    assert list(exc_info.value.offenders.index) == ["s1", "s2"]
    assert list(exc_info.value.offenders) == [2, 2]


def test_validate_accepts_valid_data_and_returns_it(compositional_table):
    kind = CompositionalTable(total=1.0, column_pattern=r"species_\w+")

    result = kind.validate(compositional_table, "taxonomy")

    assert result is compositional_table


def test_validate_rejects_rows_not_summing_to_total(compositional_table_wrong_sum):
    kind = CompositionalTable(total=1.0, column_pattern=r"species_\w+")

    with pytest.raises(RowsNotSumToTotalError) as exc_info:
        kind.validate(compositional_table_wrong_sum, "taxonomy")

    assert list(exc_info.value.offenders.index) == ["s2", "s3"]
