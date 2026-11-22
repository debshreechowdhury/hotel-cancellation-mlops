# One image, several entry points (see entrypoint.sh):
#   docker run IMAGE train       -> run the training pipeline
#   docker run IMAGE serve       -> start the FastAPI server (default)
#   docker run IMAGE preprocess  -> clean raw CSV
#   docker run IMAGE sample      -> generate synthetic data + train on it (smoke test)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# libgomp is needed by scikit-learn's HistGradientBoosting (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config.yaml entrypoint.sh ./
COPY src/ src/
COPY api/ api/
COPY scripts/ scripts/
# Trained model (if present) is baked into the image for serving.
COPY models/ models/

# Writable dirs so `docker run --user ...` with mounted volumes also works
ENV MPLCONFIGDIR=/tmp/matplotlib WANDB_DIR=/tmp
RUN chmod +x entrypoint.sh && useradd --create-home appuser \
    && mkdir -p data/raw data/processed reports \
    && chown -R appuser /app && chmod -R a+rwX /app/data /app/reports /app/models
USER appuser

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]
CMD ["serve"]
