"""FastAPI service that serves the trained cancellation model.

Run locally:
    uvicorn api.main:app --reload --port 8080
Docs (Swagger UI): http://localhost:8080/docs
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import BatchPrediction, BatchRequest, Booking, Prediction

ROOT = Path(__file__).resolve().parent.parent  # repo root (or the deployed package root)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "model.joblib"))
METADATA_PATH = Path(os.environ.get("METADATA_PATH", ROOT / "models" / "metadata.json"))
# Optional: gs://bucket/path/model.joblib — downloaded at startup if MODEL_PATH is missing
MODEL_GCS_URI = os.environ.get("MODEL_GCS_URI")

state: dict = {"model": None, "metadata": {}, "loaded": False}


def _download_gcs(uri: str, dest: Path) -> None:
    from google.cloud import storage
    bucket, blob = uri.removeprefix("gs://").split("/", 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket).blob(blob).download_to_filename(str(dest))


def load_model() -> None:
    """Idempotent: called at startup and lazily on first request (serverless hosts
    such as Azure Functions may not run the ASGI lifespan)."""
    if state["loaded"]:
        return
    state["loaded"] = True
    if not MODEL_PATH.exists() and MODEL_GCS_URI:
        print(f"[api] downloading model from {MODEL_GCS_URI}")
        _download_gcs(MODEL_GCS_URI, MODEL_PATH)
    if not MODEL_PATH.exists():
        print(f"[api] WARNING: no model at {MODEL_PATH} — /predict will return 503")
        return
    state["model"] = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    print(f"[api] loaded model {state['metadata'].get('model_version', MODEL_PATH.name)}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Hotel Cancellation Predictor",
    description="Predicts the probability that a hotel booking will be cancelled.",
    version="1.0.0",
    lifespan=lifespan,
)
# Allow browser front ends (e.g. a NextJS app) to call the API directly.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _risk_level(p: float) -> str:
    return "low" if p < 0.3 else "medium" if p < 0.6 else "high"


def _predict(bookings: list[Booking]) -> list[Prediction]:
    load_model()
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([b.model_dump() for b in bookings])
    probs = model.predict_proba(df)[:, 1]
    meta = state["metadata"]
    threshold = float(meta.get("threshold", 0.5))
    version = meta.get("model_version", "unknown")
    return [Prediction(cancel_probability=round(float(p), 4), will_cancel=bool(p >= threshold),
                       risk_level=_risk_level(float(p)), threshold=threshold,
                       model_version=version) for p in probs]


@app.get("/")
def root():
    return {"service": "hotel-cancellation-predictor", "docs": "/docs",
            "endpoints": ["/health", "/metadata", "/predict", "/predict/batch"]}


@app.get("/health")
def health():
    load_model()
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/metadata")
def metadata():
    load_model()
    meta = state["metadata"]
    return {k: meta.get(k) for k in ["model_type", "model_version", "trained_at", "threshold",
                                     "metrics", "top_features", "wandb_run_url", "features"]}


@app.post("/predict", response_model=Prediction)
def predict(booking: Booking):
    return _predict([booking])[0]


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(req: BatchRequest):
    return BatchPrediction(predictions=_predict(req.bookings))
