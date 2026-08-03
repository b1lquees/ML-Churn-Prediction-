"""
Train/serve consistency tests.

These are the regression guard for the bug where the serving transform applied
`pd.get_dummies(..., drop_first=True)` to a single-row request. With one row
there is only ever one category present per column, so drop_first deleted the
only dummy it produced and EVERY one-hot feature reached the model as 0 - the
API returned the same answer for every customer.
"""

import pandas as pd
import pytest

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.serving.inference import _serve_transform


@pytest.fixture(scope="module")
def training_artifacts(raw_telco_df):
    """Run the real training-time feature pipeline and capture its schema."""
    df = preprocess_data(raw_telco_df.copy(), target_col="Churn")
    encoded = build_features(df, target_col="Churn")
    for c in encoded.select_dtypes(include=["bool"]).columns:
        encoded[c] = encoded[c].astype(int)
    feature_cols = [c for c in encoded.columns if c != "Churn"]
    return encoded, feature_cols


def test_training_produces_expected_onehot_columns(training_artifacts):
    """drop_first at training time keeps k-1 dummies per multi-category column."""
    _, feature_cols = training_artifacts
    assert "Contract_One year" in feature_cols
    assert "Contract_Two year" in feature_cols
    # The alphabetically-first category is the dropped baseline
    assert "Contract_Month-to-month" not in feature_cols


def test_serving_matches_training_row_for_row(raw_telco_df, training_artifacts):
    """
    The serving transform must reproduce the training feature vector exactly.

    This is the core train/serve contract: feed each raw row through the serving
    path and compare against the row the training pipeline produced.
    """
    encoded, feature_cols = training_artifacts

    sample_idx = list(range(30))
    for i in sample_idx:
        raw_row = raw_telco_df.iloc[[i]].drop(columns=["Churn", "customerID"])
        served = _serve_transform(raw_row, feature_cols)

        expected = encoded.iloc[[i]][feature_cols].reset_index(drop=True)
        actual = served.reset_index(drop=True)

        mismatched = [c for c in feature_cols if actual[c].iloc[0] != expected[c].iloc[0]]
        assert not mismatched, (
            f"row {i}: serving/training disagree on {mismatched}\n"
            f"  serving : {actual[mismatched].to_dict('records')}\n"
            f"  training: {expected[mismatched].to_dict('records')}"
        )


def test_serving_actually_sets_onehot_features(training_artifacts):
    """
    A non-baseline category must produce a 1.

    Guards the specific regression: previously every one-hot feature was 0 no
    matter what the customer looked like.
    """
    _, feature_cols = training_artifacts

    customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
    }
    out = _serve_transform(pd.DataFrame([customer]), feature_cols)

    assert out["Contract_Two year"].iloc[0] == 1
    assert out["Contract_One year"].iloc[0] == 0
    assert out["InternetService_Fiber optic"].iloc[0] == 1
    assert out["PaymentMethod_Credit card (automatic)"].iloc[0] == 1
    assert out["MultipleLines_Yes"].iloc[0] == 1

    onehot_cols = [c for c in feature_cols if "_" in c]
    assert out[onehot_cols].sum(axis=1).iloc[0] > 0, "all one-hot features were zero"


def test_baseline_category_encodes_as_all_zeros(training_artifacts):
    """The dropped-first category is correctly represented by zeros."""
    _, feature_cols = training_artifacts
    customer = {"Contract": "Month-to-month", "gender": "Male", "tenure": 1}
    out = _serve_transform(pd.DataFrame([customer]), feature_cols)
    assert out["Contract_One year"].iloc[0] == 0
    assert out["Contract_Two year"].iloc[0] == 0


def test_binary_map_matches_training_encoding(raw_telco_df, training_artifacts):
    """gender/Partner/... must map to the same 0/1 on both sides."""
    encoded, feature_cols = training_artifacts
    from src.serving.inference import BINARY_MAP

    for col in BINARY_MAP:
        for i in range(10):
            raw_row = raw_telco_df.iloc[[i]].drop(columns=["Churn", "customerID"])
            served = _serve_transform(raw_row, feature_cols)
            assert served[col].iloc[0] == encoded.iloc[i][col], f"{col} differs on row {i}"


def test_serving_output_shape_and_order(training_artifacts):
    """Model input must be exactly the training columns, in training order."""
    _, feature_cols = training_artifacts
    out = _serve_transform(pd.DataFrame([{"gender": "Male", "tenure": 5}]), feature_cols)
    assert list(out.columns) == feature_cols
    assert len(out) == 1


def test_unknown_category_does_not_crash(training_artifacts):
    """An unseen category should fall back to the baseline, not raise."""
    _, feature_cols = training_artifacts
    out = _serve_transform(
        pd.DataFrame([{"Contract": "Weekly", "gender": "Nonbinary", "tenure": 3}]),
        feature_cols,
    )
    assert list(out.columns) == feature_cols
    assert out["Contract_One year"].iloc[0] == 0


def test_blank_total_charges_is_coerced(training_artifacts):
    """The dataset's literal ' ' blanks must become 0, not NaN or a string."""
    _, feature_cols = training_artifacts
    out = _serve_transform(
        pd.DataFrame([{"gender": "Male", "tenure": 0, "TotalCharges": " ",
                       "MonthlyCharges": 50.0}]),
        feature_cols,
    )
    assert out["TotalCharges"].iloc[0] == 0
    assert out.notna().all().all()
