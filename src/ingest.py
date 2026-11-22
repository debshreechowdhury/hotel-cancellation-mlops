"""Move data between local disk and cloud storage, and load it for training.

Supported cloud stores (config `data.source`):
  * supabase — Supabase Storage (free, no card). Needs SUPABASE_URL and SUPABASE_KEY.
  * gcs      — Google Cloud Storage. Needs `gcloud auth application-default login`.
  * local    — no cloud, read data/processed/... directly (tests / CI only).

Usage:
    python -m src.ingest upload     # local clean CSV -> cloud
    python -m src.ingest download   # cloud -> local clean CSV
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests

from src.config import get, load_config

# ----------------------------------------------------------------------------- Supabase


def _supabase_creds() -> tuple[str, dict]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise SystemExit(
            "[ingest] Set SUPABASE_URL and SUPABASE_KEY (in .env or your shell).\n"
            "Find them in Supabase -> Project Settings -> API Keys (use the *secret* key).")
    headers = {"apikey": key}
    if key.startswith("eyJ"):  # legacy JWT keys also go in the Authorization header
        headers["Authorization"] = f"Bearer {key}"
    return url, headers


def _raise_for(resp: requests.Response, what: str) -> None:
    if resp.status_code >= 400:
        raise SystemExit(f"[ingest] {what} failed: HTTP {resp.status_code} {resp.text[:300]}")


def supabase_ensure_bucket(bucket: str) -> None:
    url, headers = _supabase_creds()
    r = requests.post(f"{url}/storage/v1/bucket", headers=headers, timeout=30,
                      json={"id": bucket, "name": bucket, "public": False})
    if r.status_code in (200, 201):
        print(f"[ingest] created Supabase bucket '{bucket}'")
    elif "already exists" in r.text.lower() or "duplicate" in r.text.lower() or r.status_code == 409:
        pass
    else:
        _raise_for(r, f"creating bucket '{bucket}'")


def supabase_upload(local_path: str | Path, bucket: str, path: str) -> str:
    supabase_ensure_bucket(bucket)
    url, headers = _supabase_creds()
    with open(local_path, "rb") as f:
        r = requests.post(f"{url}/storage/v1/object/{bucket}/{path}", timeout=300, data=f,
                          headers={**headers, "x-upsert": "true", "Content-Type": "text/csv"})
    _raise_for(r, "upload")
    uri = f"supabase://{bucket}/{path}"
    print(f"[ingest] uploaded {local_path} -> {uri}")
    return uri


def supabase_download(bucket: str, path: str, local_path: str | Path) -> Path:
    url, headers = _supabase_creds()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"{url}/storage/v1/object/{bucket}/{path}", headers=headers,
                     timeout=300, stream=True)
    _raise_for(r, "download")
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print(f"[ingest] downloaded supabase://{bucket}/{path} -> {local_path}")
    return local_path


# ----------------------------------------------------------------------------- GCS


def _gcs_client():
    try:
        from google.cloud import storage
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install google-cloud-storage to use data.source=gcs") from e
    return storage.Client()


def _check_bucket(bucket: str) -> None:
    if not bucket or bucket == "YOUR_BUCKET_NAME":
        raise SystemExit("Set data.gcs_bucket in config.yaml (or the GCS_BUCKET env var) first.")


def upload_to_gcs(local_path: str | Path, bucket: str, blob: str) -> str:
    _check_bucket(bucket)
    _gcs_client().bucket(bucket).blob(blob).upload_from_filename(str(local_path))
    uri = f"gs://{bucket}/{blob}"
    print(f"[ingest] uploaded {local_path} -> {uri}")
    return uri


def download_from_gcs(bucket: str, blob: str, local_path: str | Path) -> Path:
    _check_bucket(bucket)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _gcs_client().bucket(bucket).blob(blob).download_to_filename(str(local_path))
    print(f"[ingest] downloaded gs://{bucket}/{blob} -> {local_path}")
    return local_path


# ----------------------------------------------------------------------------- pipeline entry


def upload(cfg: dict) -> str:
    source = get(cfg, "data.source")
    path = get(cfg, "data.clean_path")
    if source == "supabase":
        return supabase_upload(path, get(cfg, "data.supabase_bucket"), get(cfg, "data.supabase_path"))
    if source == "gcs":
        return upload_to_gcs(path, get(cfg, "data.gcs_bucket"), get(cfg, "data.gcs_blob"))
    raise SystemExit(f"[ingest] nothing to upload for data.source={source!r}")


def download(cfg: dict) -> Path:
    source = get(cfg, "data.source")
    path = get(cfg, "data.clean_path")
    if source == "supabase":
        return supabase_download(get(cfg, "data.supabase_bucket"), get(cfg, "data.supabase_path"), path)
    if source == "gcs":
        return download_from_gcs(get(cfg, "data.gcs_bucket"), get(cfg, "data.gcs_blob"), path)
    raise SystemExit(f"[ingest] nothing to download for data.source={source!r}")


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Data-ingestion step of the pipeline: fetch from the configured source."""
    source = get(cfg, "data.source", "supabase")
    local_path = Path(get(cfg, "data.clean_path"))
    if source in ("supabase", "gcs"):
        download(cfg)
    elif source == "local":
        if not local_path.exists():
            raise SystemExit(f"[ingest] data.source=local but {local_path} does not exist")
        print(f"[ingest] reading local file {local_path}")
    else:
        raise ValueError(f"Unknown data.source: {source!r} (use supabase, gcs or local)")
    df = pd.read_csv(local_path)
    print(f"[ingest] loaded {len(df):,} rows x {df.shape[1]} cols")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    upload(cfg) if args.action == "upload" else download(cfg)


if __name__ == "__main__":
    main()
