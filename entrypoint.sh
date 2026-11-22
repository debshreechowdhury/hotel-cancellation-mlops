#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-serve}"
shift || true

case "$cmd" in
  train)
    exec python -m src.train "$@"
    ;;
  preprocess)
    exec python -m src.preprocess "$@"
    ;;
  serve)
    exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8080}"
    ;;
  sample)
    # Smoke test with synthetic data: no cloud credentials or W&B key needed
    python scripts/make_sample_data.py --rows 3000 --out data/raw/sample.csv
    python -m src.preprocess --raw data/raw/sample.csv --out data/processed/sample_clean.csv
    exec python -m src.train --set data.source=local \
      --set data.clean_path=data/processed/sample_clean.csv \
      --set tracking.backend=none "$@"
    ;;
  *)
    exec "$cmd" "$@"
    ;;
esac
