"""Move data between local disk and Google Cloud Storage, and load it for training.

Usage:
    python -m src.ingest upload     # local clean CSV  -> gs://<bucket>/<blob>
    python -m src.ingest download   # gs://<bucket>/<blob> -> local clean CSV

Auth: run `gcloud auth application-default login` once on your machine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import get, load_config


def _client():
    try:
        from google.cloud import storage
    except ImportError as e:  # pragma: no cover
        raise ImportError("google-cloud-storage is not installed: pip install -r requirements.txt") from e
    return storage.Client()


def _check_bucket(bucket: str) -> None:
    if not bucket or bucket == "YOUR_BUCKET_NAME":
        raise SystemExit("Set data.gcs_bucket in config.yaml (or the GCS_BUCKET env var) first.")


def upload_to_gcs(local_path: str | Path, bucket: str, blob: str) -> str:
    _check_bucket(bucket)
    _client().bucket(bucket).blob(blob).upload_from_filename(str(local_path))
    uri = f"gs://{bucket}/{blob}"
    print(f"[ingest] uploaded {local_path} -> {uri}")
    return uri


def download_from_gcs(bucket: str, blob: str, local_path: str | Path) -> Path:
    _check_bucket(bucket)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().bucket(bucket).blob(blob).download_to_filename(str(local_path))
    print(f"[ingest] downloaded gs://{bucket}/{blob} -> {local_path}")
    return local_path


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Data-ingestion step of the pipeline: fetch from the configured source."""
    source = get(cfg, "data.source", "gcs")
    local_path = Path(get(cfg, "data.clean_path"))
    if source == "gcs":
        download_from_gcs(get(cfg, "data.gcs_bucket"), get(cfg, "data.gcs_blob"), local_path)
    elif source == "local":
        if not local_path.exists():
            raise SystemExit(f"[ingest] data.source=local but {local_path} does not exist")
        print(f"[ingest] reading local file {local_path}")
    else:
        raise ValueError(f"Unknown data.source: {source!r} (use 'gcs' or 'local')")
    df = pd.read_csv(local_path)
    print(f"[ingest] loaded {len(df):,} rows x {df.shape[1]} cols")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    bucket, blob = get(cfg, "data.gcs_bucket"), get(cfg, "data.gcs_blob")
    path = get(cfg, "data.clean_path")
    if args.action == "upload":
        upload_to_gcs(path, bucket, blob)
    else:
        download_from_gcs(bucket, blob, path)


if __name__ == "__main__":
    main()
