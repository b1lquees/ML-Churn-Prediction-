# 1. Use the official lightweight Python base image.
#    Pinned to 3.11 because the serving model was pickled under Python 3.11.
FROM python:3.11-slim

# 2. Create the runtime user up front.
#    Hugging Face Spaces runs the container as UID 1000, so the files have to be
#    owned by that UID. Creating the user before any COPY lets us set ownership
#    inline with --chown below, instead of a recursive chown afterwards: HF's
#    docs warn that `chown -R` rewrites every file into a new layer and can
#    balloon the image.
RUN useradd --create-home --uid 1000 appuser

# 3. Set working directory, owned by that user so the app can write at runtime
WORKDIR /app
RUN chown appuser:appuser /app

# 4. Copy only the dependency file first (for Docker layer caching)
COPY requirements.txt .

# 5. Install Python dependencies.
#    Only the serving runtime - training tools live in requirements-dev.txt.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# 6. Copy the project source (see .dockerignore for what is left out)
COPY --chown=appuser:appuser src/ /app/src/

# 7. Publish the bundled MLflow run to the flat /app/model path that the serving
#    code looks at first. ARG lets you build against a different run without
#    editing this file:
#        docker build --build-arg MODEL_RUN_ID=<other-run-id> -t telco-churn-app .
ARG MODEL_RUN_ID=3b1a41221fc44548aed629fa42b762e0
COPY --chown=appuser:appuser src/serving/model/${MODEL_RUN_ID}/artifacts/model /app/model
COPY --chown=appuser:appuser src/serving/model/${MODEL_RUN_ID}/artifacts/feature_columns.txt /app/model/feature_columns.txt

# 8. Environment
#    PYTHONPATH=/app/src lets modules import as `serving.*` as well as `src.*`.
#    PYTHONUNBUFFERED=1 streams logs out in real time instead of buffering them.
#    PYTHONIOENCODING=utf-8 keeps a stray non-ASCII character out of the way of
#      the logs on hosts whose console is not UTF-8.
#    MODEL_DIR is read by src/serving/inference.py.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PYTHONIOENCODING=utf-8 \
    MODEL_DIR=/app/model \
    HOME=/home/appuser

# 9. Drop root
USER appuser

# 10. Expose FastAPI port.
#     Hugging Face Spaces defaults to 7860 - the `app_port: 8000` line in the
#     README front matter points it here instead.
EXPOSE 8000

# 11. Container-level health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request,sys; p=os.getenv('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=4).status==200 else 1)"

# 12. Run the FastAPI app using uvicorn.
#     Shell form so ${PORT} expands at runtime: Render (and most PaaS hosts)
#     inject PORT and require the app to listen on it. Falls back to 8000, which
#     is what the local `docker run -p 8000:8000` and EXPOSE above assume.
CMD ["sh", "-c", "exec python -m uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
