
#Backwards-compatible entry point.
#`src/app/main.py` is the single definition of the FastAPI + Gradio application.
#It just re-exports the real app so that uvicorn src.app.app:app
#keeps working and can never disagree with `src.app.main:app`.

import os
import sys
# Allow "uvicorn src.app.app:app" to resolve `src.*` when started from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.app.main import app, demo, CustomerData, gradio_interface  # noqa: F401,E402

__all__ = ["app", "demo", "CustomerData", "gradio_interface"]
