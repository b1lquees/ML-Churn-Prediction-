"""Shared pytest fixtures."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def raw_telco_df() -> pd.DataFrame:
    """
    A synthetic frame with the exact schema, categories and quirks of the real
    Telco Customer Churn dataset - including the literal " " blanks that the
    real file carries in TotalCharges for tenure-0 customers.

    Generated rather than committed so the tests run on a fresh clone, where
    data/raw/ is gitignored and therefore empty.
    """
    n = 400
    rng = np.random.default_rng(0)

    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def internet_dependent():
        """Service columns read 'No internet service' when there is no internet."""
        return np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))

    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18.0, 119.0, n), 2)
    total = np.round(monthly * np.maximum(tenure, 1), 2).astype(object)
    total[tenure == 0] = " "  # the real dataset's blank cells

    return pd.DataFrame(
        {
            "customerID": [f"{i:04d}-ABCDE" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(
                phone == "No", "No phone service", rng.choice(["Yes", "No"], n)
            ),
            "InternetService": internet,
            "OnlineSecurity": internet_dependent(),
            "OnlineBackup": internet_dependent(),
            "DeviceProtection": internet_dependent(),
            "TechSupport": internet_dependent(),
            "StreamingTV": internet_dependent(),
            "StreamingMovies": internet_dependent(),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
        }
    )
