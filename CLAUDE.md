# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training Pipeline
```bash
# Run the complete ML training pipeline
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn

# Prepare processed data only
python scripts/prepare_processed_data.py
```

### Testing
```bash
# Automated test suite (this is what CI runs)
pytest

# Single file / pattern
pytest tests/test_train_serve_parity.py
pytest -k "constant_classifier"

# Manual/exploratory scripts (require data or a running server)
python scripts/test_pipeline_phase1_data_features.py   # data + feature engineering
python scripts/test_pipeline_phase2_modeling.py        # Optuna hyperparameter search
python scripts/test_fastapi.py                         # smoke-test a running server
```

### Local Development
```bash
# Run the FastAPI + Gradio application locally
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000

# Alternative app entry point
python -m uvicorn src.app.app:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build and run the containerized application
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app

# Package a different MLflow run instead of the bundled default
docker build --build-arg MODEL_RUN_ID=<run-id> -t telco-churn-app .
```

### Dependencies
`requirements.txt` is the **serving runtime only** (what the Docker image installs).
`requirements-dev.txt` pulls it in via `-r` and adds Great Expectations, Optuna,
Jupyter and pytest. Install both for local development.

## Architecture Overview

### ML Pipeline Flow
This project implements a complete MLOps pipeline with two distinct phases:

**Training Pipeline** (`scripts/run_pipeline.py`):
1. **Data Loading** → **Data Validation** (Great Expectations) → **Preprocessing** → **Feature Engineering** → **XGBoost Training** → **MLflow Logging**
2. All artifacts (model, feature columns, preprocessing logic) are stored in MLflow for reproducibility

**Serving Pipeline** (`src/app/main.py` + `src/serving/inference.py`):
1. **FastAPI REST API** (`/predict` endpoint) + **Gradio Web UI** (`/ui` endpoint) → **MLflow Model Loading** → **Feature Transformation** → **Prediction**
2. Feature processing mirrors training-time transformations for consistency

### MLflow Integration Patterns
- **Experiment Name**: "Telco Churn" (default, can be overridden)
- **Tracking URI**: File-based at `{project_root}/mlruns`
- **Logged Artifacts**: `model/`, `feature_columns.txt`, `preprocessing.pkl`
- **Tracked Metrics**: precision, recall, f1, roc_auc, train_time, pred_time, data_quality_pass
- **Parameters**: model type, threshold (default 0.35), test_size (default 0.2)

### Feature Engineering Consistency
Critical pattern: Training and serving must use identical feature transformations.

**Training** (`src/features/build_features.py`):
- Binary features (Yes/No, Male/Female) → deterministic 0/1 mapping
- Multi-category features → one-hot encoding with `drop_first=True`
- Boolean columns → integers

**Serving** (`src/serving/inference.py`):
- Uses fixed `BINARY_MAP` dictionary for consistent binary encoding
- Applies `pd.get_dummies()` **without** `drop_first` — see the warning below
- Feature alignment via the training feature schema loaded at startup

> **Do not add `drop_first=True` to the serving transform.** A request is a
> single row, so only one category is ever present per column; `drop_first`
> would drop the only dummy produced and send an all-zero one-hot vector to the
> model, making every prediction identical. Training's dropped baseline is
> reproduced instead by the `reindex` onto the saved feature columns, which keeps
> only the k-1 columns training kept. `tests/test_train_serve_parity.py` locks
> this behavior in — it asserts serving and training agree row for row.

### Model Loading and Serving
- **Serving does NOT import mlflow.** It still reads an MLflow run directory (the `MLmodel` file identifies one) but loads `model.pkl` inside it directly, because the sklearn flavor is a plain pickle and the model was logged without a signature — verified identical predictions across a sweep of customers. This keeps mlflow/sqlalchemy/alembic/flask/pyarrow out of the runtime image (~235 MB). `mlflow` lives in `requirements-dev.txt`; do not add it back to `requirements.txt`.
- **Resolution order**: `$MODEL_DIR` → `/app/model` → bundled `src/serving/model/<run>/artifacts/model` → newest `mlruns/` run. The same code therefore works in the container and on a fresh clone.
- **Lazy loading**: the model loads on the first prediction, not at import, so a missing model can't stop the app from answering health checks.
- **Feature Order**: Enforced using `feature_columns.txt`, searched inside the model dir, then the run artifacts root, then `artifacts/`.
- **Prediction Format**: Returns "Likely to churn" or "Not likely to churn" strings

### Data Validation
- **Tool**: Great Expectations **1.x** with a custom suite (Data Context → Data Source → Batch Definition → Validation Definition). The 0.x `ge.dataset.PandasDataset` helper does not exist in this line — don't reintroduce it.
- **Location**: `src/utils/validate_data.py`
- **Checks**: CustomerID presence, categorical value sets, numeric ranges for tenure/charges, TotalCharges ≥ MonthlyCharges
- **TotalCharges**: validated on an internally-coerced copy, because the raw CSV stores it as text with literal `" "` blanks for tenure-0 customers. The caller's DataFrame is never mutated.
- **Integration**: Results logged to MLflow as `data_quality_pass` metric

### Docker Containerization
- **Base Image**: `python:3.11-slim`
- **Key Setting**: `PYTHONPATH=/app/src` for proper module imports
- **Model Artifacts**: Specific MLflow run copied to `/app/model` during build
- **Serving**: uvicorn with FastAPI app on port 8000

### CI/CD Pipeline
- **Trigger**: Push to main branch, and pull requests
- **Actions**: `pytest` → build the Docker image → run the container and smoke-test `/health` and `/predict`
- **No registry credentials**: nothing is pushed to Docker Hub. The image build exists so a broken Dockerfile fails CI rather than the Space build.
- **Deployment**: Render, as a Docker web service, configured by `render.yaml` (Blueprint). Render builds the image from the repo; `autoDeploy` rebuilds on push to main.
- **Render specifics**: Render injects `PORT`, so the Dockerfile `CMD` uses shell form and reads `${PORT:-8000}` — keep it shell form or the variable will not expand. `healthCheckPath` is `/health`, which answers before the model loads, so a slow first load cannot fail a deploy. The free instance sleeps after ~15 min idle.
- **Hugging Face Spaces now requires a paid PRO plan** for Gradio and Docker Spaces; only Static Spaces are free. The `sdk: docker` / `app_port: 8000` front matter in `README.md` is kept so the repo still works as a Space on a paid plan, but it is not the deployment target.
- **Dockerfile conventions Spaces requires**: the runtime user is UID 1000 (Spaces runs the container as that UID), and ownership is set inline with `COPY --chown` rather than a trailing `chown -R`, which HF warns rewrites every file into a new layer and bloats the image. Create the user before any `COPY`.

## Key Implementation Details

### XGBoost Model Configuration
Optimized hyperparameters are hardcoded in `scripts/run_pipeline.py`:
- `n_estimators=301`, `learning_rate=0.034`, `max_depth=7`
- `scale_pos_weight` calculated dynamically for class imbalance handling

### API Endpoints
- `GET /` - Redirects to `/ui/`. The site root is the front door on a hosted Space, so it must not return JSON.
- `GET /health` - Health check returning `{"status": "ok"}`. The Dockerfile HEALTHCHECK and CI smoke test both hit this path — change one and change all three.
- `POST /predict` - Accepts `CustomerData` Pydantic model with 19 customer attributes. Returns 422 on invalid input and 500 on inference failure (never a 200 with an error body).
- `/ui` - Gradio interface mounted via `gr.mount_gradio_app()`

`CustomerData` must stay in sync with the trained feature set — `SeniorCitizen`
is one of the model's features, so it belongs in both the API schema and the
Gradio form. A field the model was trained on but the schema omits silently
reaches the model as 0 for every request.

### File System Layout
- `data/raw/Telco-Customer-Churn.csv` - Source dataset, committed (~1 MB) so the pipeline runs on a fresh clone. Everything else under `data/raw/` is ignored.
- `data/processed/` - Derived output, gitignored; regenerated by the pipeline
- `mlruns/` - MLflow experiment tracking database
- `artifacts/` - Shared preprocessing artifacts (`feature_columns.json`, `preprocessing.pkl`)
- `src/serving/model/` - Bundled MLflow run, committed so the API works on a fresh clone
- `tests/` - pytest suite

### Development Notes
- `pytest` is the test suite; `scripts/test_*.py` are manual/exploratory scripts, not pytest tests
- MLflow UI can be accessed with: `mlflow ui --backend-store-uri file:./mlruns`
- The project uses file-based MLflow tracking (not a tracking server)
- Model serving expects exact feature column order from training time
- Entry-point scripts call `configure_console()` to force UTF-8 stdout; without it the emoji log lines raise `UnicodeEncodeError` on a Windows cp1252 console
- Pinned versions of numpy/pandas/scikit-learn/xgboost must match the pickled model in `src/serving/model/`; bumping one without retraining risks unpickling failures
- `gradio` and `huggingface_hub` are pinned together — gradio 4.x imports `HfFolder`, which huggingface_hub removed in 1.0