"""
End-to-end API tests against the bundled serving model.

These exercise the same code path the container runs, so they catch model
loading and feature alignment problems before deployment.
"""

import re

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.app.main import app  # noqa: E402

client = TestClient(app)

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 95.0,
}

LOW_RISK = dict(
    HIGH_RISK,
    Contract="Two year", InternetService="DSL", tenure=60,
    PaymentMethod="Credit card (automatic)", MonthlyCharges=45.0,
    TotalCharges=2700.0, OnlineSecurity="Yes", TechSupport="Yes",
)

VALID_PREDICTIONS = {"Likely to churn", "Not likely to churn"}


def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_redirects_to_ui():
    """The site root is the Gradio UI, so a Space visitor lands on the app."""
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/ui/"


def test_root_reaches_the_ui_when_followed():
    r = client.get("/", follow_redirects=True)
    assert r.status_code == 200


def test_predict_returns_a_valid_label():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200, r.text
    assert r.json()["prediction"] in VALID_PREDICTIONS


def test_model_is_not_a_constant_classifier():
    """
    The regression guard for the one-hot bug.

    When serving dropped every dummy, the model saw an identical all-zero vector
    for every request and returned one fixed answer. A clearly high-risk and a
    clearly low-risk customer must not come back the same.
    """
    high = client.post("/predict", json=HIGH_RISK).json()["prediction"]
    low = client.post("/predict", json=LOW_RISK).json()["prediction"]
    assert high != low, f"both profiles returned {high!r} - features are not reaching the model"
    assert high == "Likely to churn"
    assert low == "Not likely to churn"


def test_senior_citizen_is_accepted_and_used():
    """SeniorCitizen is a trained feature; the API must accept it."""
    r = client.post("/predict", json=dict(HIGH_RISK, SeniorCitizen=1))
    assert r.status_code == 200, r.text
    assert r.json()["prediction"] in VALID_PREDICTIONS


def test_senior_citizen_defaults_when_omitted():
    payload = {k: v for k, v in HIGH_RISK.items() if k != "SeniorCitizen"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text


def test_missing_required_field_returns_422():
    r = client.post("/predict", json={"gender": "Female"})
    assert r.status_code == 422


def test_bad_numeric_type_returns_422():
    r = client.post("/predict", json=dict(HIGH_RISK, tenure="not-a-number"))
    assert r.status_code == 422


def test_gradio_ui_is_mounted():
    r = client.get("/ui/")
    assert r.status_code == 200


def test_gradio_ui_serves_its_frontend_config():
    """
    The /ui page must carry the Gradio bootstrap config, not just return 200.

    A bare status check passes even when the page is a shell that never boots,
    so assert the config blob the frontend needs is actually embedded.
    """
    html = client.get("/ui/").text
    assert "window.gradio_config" in html
    match = re.search(r'"root"\s*:\s*"([^"]*)"', html)
    assert match, "no root in the embedded Gradio config"
