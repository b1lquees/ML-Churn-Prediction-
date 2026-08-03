# Telco Churn – End-to-End ML Project
Predicting telecom customer churn, from data prep and modeling through to a REST API + web UI, containerized and deployable to Render in one step.

## Problem solved & benefits
- **Faster decisions:** Predicts which customers are likely to churn so teams can act before they leave.
- **Operationalized ML:** Model is accessible via a REST API and a simple UI; anyone can test it without notebooks.
- **Repeatable delivery:** CI/CD + containers mean every change can be rebuilt, tested, and redeployed in a consistent way.
- **Traceable experiments:** MLflow tracks runs, metrics, and artifacts for reproducibility and auditing.

## Quick start

### 1. Install

Python **3.11** is recommended (the bundled model was pickled under 3.11).

```bash
python -m venv .venv
Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

`requirements.txt` is the serving runtime (what the Docker image installs).
`requirements-dev.txt` pulls that in via `-r` and adds the training, notebook and
test tooling, so the one command above gets you everything.

### 2. Run the app

A trained model is bundled in `src/serving/model/`, so the API works immediately
after install — no training required.

```bash
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

- `GET  /` — redirects to the Gradio UI
- `GET  /health` — health check → `{"status": "ok"}`
- `POST /predict` — JSON in, churn prediction out
- `/ui` — Gradio web interface
- `/docs` — interactive OpenAPI docs

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":1,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":95.0,"TotalCharges":95.0}'
```

### 3. Run the tests

```bash
pytest
```

### 4. (Optional) Retrain

The source dataset is committed at `data/raw/Telco-Customer-Churn.csv` (the IBM
*Telco Customer Churn* sample, 7,043 customers — also on
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)), so
retraining needs no extra downloads:

```bash
python scripts/run_pipeline.py
```

Anything written to `data/processed/` is derived output and stays gitignored —
the pipeline regenerates it. Point at a different file with `--input` or the
`DATA_PATH` environment variable.

Typical run on this dataset: ROC AUC ≈ 0.84, recall ≈ 0.83 at the default 0.35
threshold (recall is deliberately favoured — missing a churner costs more than a
false alarm).

Inspect the runs with:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### 5. Docker

```bash
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
```

To package a model you just trained instead of the bundled one:

```bash
docker build --build-arg MODEL_RUN_ID=<your-mlflow-run-id> -t telco-churn-app .
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `/app/model` | Where the serving code loads the model from. Falls back to the bundled model, then the newest local `mlruns/` run. |
| `PORT` | `8000` | Port uvicorn binds. Render injects this; the container reads it. |
| `DATA_PATH` | `data/raw/Telco-Customer-Churn.csv` | Input CSV for the data-prep scripts. |
| `BASE_URL` | `http://127.0.0.1:8000` | Target for `scripts/test_fastapi.py`. |

## What I built

- **Data & Modeling:** Feature engineering + XGBoost classifier; experiments logged to MLflow.
- **Data quality gate:** Great Expectations suite runs before training and blocks bad data.
- **Model tracking:** Runs, metrics, and the serialized model logged under a named MLflow experiment.
- **Inference service:** FastAPI app exposing `/predict` (POST) and a `/health` check.
- **Web UI:** Gradio interface at `/ui` (the site root redirects there).
- **Tests:** pytest suite covering train/serve feature parity, the validation gate, and the API.
- **Containerization:** Docker image with uvicorn entrypoint (`src.app.main:app`) listening on port 8000, running as a non-root user.
- **CI:** GitHub Actions runs the test suite and builds the image on every push and pull request.
- **Hosting:** Deployed to Render as a Docker web service — the same image runs locally and in production.
- **Lean runtime:** serving does not import MLflow. The model is logged by MLflow at training time but loaded straight from its pickle at serving time, cutting ~235 MB of dependencies out of the production image.

## Project layout

```
data/raw/            # source dataset (committed)
data/processed/      # derived output written by the pipeline (gitignored)
notebooks/EDA.ipynb  # exploratory analysis
scripts/             # pipeline entry points and manual test scripts
src/data/            # loading + preprocessing
src/features/        # feature engineering (training side)
src/serving/         # inference (serving side) + bundled MLflow model
src/utils/           # data validation + console encoding helper
src/app/main.py      # FastAPI + Gradio application
tests/               # pytest suite
Dockerfile           # serving image
render.yaml          # Render Blueprint (deployment config)
```

## Render
| Path | What |
|---|---|
| `/` | redirects to the UI |
| `/ui` | Gradio interface |
| `/predict` | REST endpoint (POST) |
| `/docs` | interactive API docs |
| `/health` | health check |

### Notes

- **The free instance sleeps** after ~15 minutes idle and takes about a minute to
  wake. 
- **`PORT` is injected by Render**; the container's `CMD` reads `${PORT:-8000}`,
  so the same image runs unchanged locally and on Render.
- **`/health` is the health check path on purpose** — it responds before the
  model is loaded, so a slow first load can't fail the deploy.
- **The model is committed**, so there is no external storage or secret to set up.

### Other hosts

- **Hugging Face Spaces** requires a **paid PRO plan** for Gradio and Docker
  Spaces; only Static Spaces are free, and those cannot run Python. The
  `sdk: docker` / `app_port: 8000` front matter at the top of this file is left
  in place so the repo works as a Space if you ever subscribe.
- **Google Cloud Run** works with this same Dockerfile and has a generous
  always-free allowance, but requires a card on the account.

### Quick temporary demo (no hosting)

To share a link straight from your machine without deploying anywhere:

```bash
python -c "from src.app.main import demo; demo.launch(share=True)"
```

That prints a public URL that lasts about 72 hours and only works while your
machine stays awake. Useful for a quick show-and-tell, not for deployment.

### CI

GitHub Actions runs the test suite, builds the Docker image, boots the container
and smoke-tests `/health` and `/predict` on every push and pull request. No
registry credentials are needed — Render builds the image itself from the repo.
