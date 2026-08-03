"""Tests for the Great Expectations data quality gate."""

import pandas as pd
import pytest

from src.data.preprocess import preprocess_data
from src.utils.validate_data import validate_telco_data


def test_clean_data_passes(raw_telco_df):
    """A well-formed frame - blanks in TotalCharges included - must pass."""
    ok, failed = validate_telco_data(raw_telco_df)
    assert ok, f"clean data failed validation: {failed}"
    assert failed == []


def test_validation_does_not_mutate_caller_frame(raw_telco_df):
    """Validation coerces TotalCharges internally; the caller's frame is its own."""
    before = raw_telco_df["TotalCharges"].copy()
    validate_telco_data(raw_telco_df)
    pd.testing.assert_series_equal(raw_telco_df["TotalCharges"], before)


@pytest.mark.parametrize(
    "column,bad_value,expected_failure",
    [
        ("gender", "Unknown", "expect_column_values_to_be_in_set"),
        ("Contract", "Weekly", "expect_column_values_to_be_in_set"),
        ("InternetService", "Satellite", "expect_column_values_to_be_in_set"),
        ("Partner", "Maybe", "expect_column_values_to_be_in_set"),
        ("tenure", -5, "expect_column_values_to_be_between"),
        ("MonthlyCharges", 5000, "expect_column_values_to_be_between"),
        ("customerID", None, "expect_column_values_to_not_be_null"),
    ],
)
def test_corrupt_values_are_caught(raw_telco_df, column, bad_value, expected_failure):
    df = raw_telco_df.copy()
    df.loc[df.index[0], column] = bad_value
    ok, failed = validate_telco_data(df)
    assert not ok, f"corrupting {column} with {bad_value!r} should fail validation"
    assert expected_failure in failed


def test_missing_required_column_is_caught(raw_telco_df):
    df = raw_telco_df.drop(columns=["MonthlyCharges"])
    ok, failed = validate_telco_data(df)
    assert not ok
    assert "expect_column_to_exist" in failed


def test_preprocess_runs_after_validation(raw_telco_df):
    """The pipeline order (validate -> preprocess) must produce a clean frame."""
    assert validate_telco_data(raw_telco_df)[0]
    out = preprocess_data(raw_telco_df.copy(), target_col="Churn")
    assert "customerID" not in out.columns          # ID dropped
    assert set(out["Churn"].unique()) <= {0, 1}     # target encoded
    assert out["TotalCharges"].dtype.kind == "f"    # coerced to float
    assert out.select_dtypes(include=["number"]).isna().sum().sum() == 0
