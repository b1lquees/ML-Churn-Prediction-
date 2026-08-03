"""
Smoke test for the running FastAPI service.

Start the app first:
    python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000

Then:
    python scripts/test_fastapi.py
    BASE_URL=http://my-alb-dns python scripts/test_fastapi.py
"""

import os
import sys

import requests

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# A high-risk profile: month-to-month, fiber, electronic check, brand-new customer
HIGH_RISK = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 95.0,
}

# A low-risk profile: two-year contract, long tenure, automatic payment
LOW_RISK = dict(
    HIGH_RISK,
    Contract="Two year",
    InternetService="DSL",
    PaymentMethod="Credit card (automatic)",
    tenure=60,
    MonthlyCharges=45.0,
    TotalCharges=2700.0,
    OnlineSecurity="Yes",
    TechSupport="Yes",
)

failures = []


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}{'' if condition else f'  -> {detail}'}")
    if not condition:
        failures.append(name)


print(f"Testing {BASE_URL}\n")

#health check
try:
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    check("GET /health returns 200", r.status_code == 200, f"got {r.status_code}")
    check("GET /health reports ok", r.json().get("status") == "ok", r.text[:200])
except requests.RequestException as e:
    print(f"ERROR: could not reach {BASE_URL} - is the app running?  ({e})")
    sys.exit(1)

#predictions
results = {}
for label, payload in [("high risk", HIGH_RISK), ("low risk", LOW_RISK)]:
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
    check(f"POST /predict ({label}) returns 200", r.status_code == 200, r.text[:300])
    if r.status_code == 200:
        body = r.json()
        check(f"POST /predict ({label}) has a prediction", "prediction" in body, r.text[:300])
        results[label] = body.get("prediction")
        print(f"     {label}: {body.get('prediction')}")

# The model must not be a constant classifier - this is the regression guard for
# the one-hot encoding bug that made every request return the same answer.
if len(results) == 2:
    check(
        "predictions differ between high-risk and low-risk profiles",
        results["high risk"] != results["low risk"],
        f"both returned {results['high risk']!r}",
    )

# validation error handling
r = requests.post(f"{BASE_URL}/predict", json={"gender": "Female"}, timeout=10)
check("POST /predict with missing fields returns 422", r.status_code == 422, f"got {r.status_code}")

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("All API checks passed.")
