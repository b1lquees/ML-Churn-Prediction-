"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Locate and load an MLflow-logged model plus its feature metadata
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

NOTE ON MLFLOW: training logs the model with MLflow, and this module still reads
an MLflow run directory (it uses the MLmodel descriptor to recognise one). But it
loads the pickle inside DIRECTLY rather than through `mlflow.pyfunc.load_model`,
so mlflow is NOT a serving dependency. The MLflow sklearn flavor is a plain
pickle of the estimator, and pyfunc adds nothing here because the model was
logged without a signature - verified identical predictions across a sweep of
customers. Dropping it removes mlflow plus sqlalchemy/alembic/flask/pyarrow from
the runtime image, which is what lets this fit a small free-tier instance.

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- One-hot encodes WITHOUT drop_first, reproducing training's dropped baseline
  through the reindex onto the saved feature schema (see _serve_transform)
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Model resolution order (first hit wins):
1. $MODEL_DIR                                  - explicit override
2. /app/model                                  - path baked in by the Dockerfile
3. src/serving/model/<run_id>/artifacts/model  - model bundled in this repo
4. mlruns/<exp>/<run>/artifacts/model          - most recent local training run

Loading is LAZY: the model is resolved on the first prediction, not at import.
That keeps a missing/broken model from taking down the whole web app, so `/`
still answers health checks and the error surfaces per request instead.
"""

import os
import glob
import pickle
import threading
from pathlib import Path

import pandas as pd

# Repo root: .../src/serving/inference.py -> up 3 levels
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The bundled run the Dockerfile publishes to /app/model. Kept in sync with the
# MODEL_RUN_ID build arg so local development serves the same model the
# container does; more than one run is bundled, and picking by timestamp made
# which one you got depend on checkout order.
BUNDLED_RUN_ID = "3b1a41221fc44548aed629fa42b762e0"

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# (see src/features/build_features.py::_map_binary_series).
# Any changes here will cause train/serve skew and degrade model performance.
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def _candidate_model_dirs():
    """Yield every place a serving model might live, best candidate first."""
    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        yield Path(env_dir)

    # Path baked into the Docker image at build time
    yield Path("/app/model")

    # Model bundled in this repository (works straight after `git clone`).
    # The Dockerfile's run comes first; any other bundled run is a deterministic
    # by-name fallback.
    bundled_root = PROJECT_ROOT / "src" / "serving" / "model"
    yield bundled_root / BUNDLED_RUN_ID / "artifacts" / "model"
    for path in sorted(bundled_root.glob("*/artifacts/model")):
        yield path

    # Most recent run from local MLflow tracking (i.e. after running the pipeline)
    local_runs = sorted(
        glob.glob(str(PROJECT_ROOT / "mlruns" / "*" / "*" / "artifacts" / "model")),
        key=os.path.getmtime,
        reverse=True,
    )
    for p in local_runs:
        yield Path(p)


def _resolve_model_dir() -> Path:
    """
    Return the first candidate directory that actually holds an MLflow model.

    An MLflow model directory is identified by its MLmodel descriptor file.
    """
    checked = []
    for candidate in _candidate_model_dirs():
        checked.append(str(candidate))
        if (candidate / "MLmodel").is_file():
            return candidate
    raise FileNotFoundError(
        "No MLflow model found. Set MODEL_DIR, or train one with "
        "`python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv`. "
        f"Looked in: {checked}"
    )


def _load_feature_cols(model_dir: Path) -> list:
    """
    Load the exact feature column order used during training.

    The schema is written next to the model by the training pipeline, but older
    runs put it one level up in the run's artifacts/ folder, and the local dev
    flow writes a JSON copy under artifacts/. Check all three.
    """
    txt_candidates = [
        model_dir / "feature_columns.txt",           # inside the model dir
        model_dir.parent / "feature_columns.txt",    # run artifacts root
        PROJECT_ROOT / "artifacts" / "feature_columns.txt",
    ]
    for path in txt_candidates:
        if path.is_file():
            cols = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if cols:
                return cols

    json_path = PROJECT_ROOT / "artifacts" / "feature_columns.json"
    if json_path.is_file():
        import json
        cols = json.loads(json_path.read_text(encoding="utf-8"))
        if cols:
            return cols

    raise FileNotFoundError(
        "Could not find the training feature schema (feature_columns.txt). "
        f"Looked in: {[str(p) for p in txt_candidates] + [str(json_path)]}"
    )


def _load_pickled_model(model_dir: Path):
    """
    Load the estimator pickled inside an MLflow model directory.

    The sklearn flavor writes the estimator to model.pkl with cloudpickle.
    cloudpickle emits a standard pickle stream, so stdlib pickle can read it as
    long as the estimator's class is importable - which XGBClassifier is. We
    still prefer cloudpickle when it is installed, and fall back to pickle.
    """
    pkl = model_dir / "model.pkl"
    if not pkl.is_file():
        others = sorted(model_dir.glob("*.pkl"))
        if not others:
            raise FileNotFoundError(f"No .pkl estimator found in {model_dir}")
        pkl = others[0]

    with open(pkl, "rb") as f:
        try:
            import cloudpickle
            return cloudpickle.load(f)
        except ImportError:
            f.seek(0)
            return pickle.load(f)


# Lazily-populated module state, guarded so concurrent requests load only once.
_model = None
_feature_cols = None
_model_dir = None
_load_lock = threading.Lock()


def get_model():
    """Load (once) and return the model, its feature schema, and its directory."""
    global _model, _feature_cols, _model_dir
    if _model is None:
        with _load_lock:
            if _model is None:  # re-check: another thread may have won the race
                model_dir = _resolve_model_dir()
                feature_cols = _load_feature_cols(model_dir)
                # Assign the model last: if loading raises, state stays unset
                # and the next request retries instead of serving a half-init.
                _feature_cols = feature_cols
                _model_dir = model_dir
                _model = _load_pickled_model(model_dir)
                print(f"Model loaded from {model_dir}")
                print(f"Loaded {len(feature_cols)} feature columns from training")
    return _model, _feature_cols, _model_dir


def _serve_transform(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.

    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.

    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features
    4. Convert boolean columns to integers
    5. Align features with training schema and order

    Args:
        df: Single-row DataFrame with raw customer data
        feature_cols: Exact feature order captured at training time

    Returns:
        DataFrame with features transformed and ordered for model input

    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()

    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()

    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)

    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )

    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding. NOTE: drop_first is deliberately NOT used here.
        # A single-row request has one value per column, so drop_first would drop
        # that one dummy and encode every customer as all-zeros. Training's
        # drop_first baseline is instead reproduced by the reindex in STEP 5,
        # which keeps only the columns training kept.
        df = pd.get_dummies(df, columns=obj_cols)

    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training.
    # Missing features get filled with 0, extra features are dropped.
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Guard against object dtypes sneaking through into the model
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.

    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.

    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string

    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema

    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)

    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """
    model, feature_cols, _ = get_model()

    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])

    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df, feature_cols)

    # === STEP 3: Generate Model Prediction ===
    preds = model.predict(df_enc)

    # Normalize prediction output to consistent format
    if hasattr(preds, "tolist"):
        preds = preds.tolist()  # Convert numpy array to list

    # Extract single prediction value (for single-row input)
    if isinstance(preds, (list, tuple)) and len(preds) == 1:
        result = preds[0]
    else:
        result = preds

    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if int(result) == 1:
        return "Likely to churn"      # High risk - needs intervention
    return "Not likely to churn"      # Low risk - maintain normal service
