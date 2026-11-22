#!/usr/bin/env bash
# Build the API image with Cloud Build and deploy it to Cloud Run.
# Prereqs: gcloud CLI logged in, a trained model in models/, billing enabled.
#
#   PROJECT_ID=my-project ./scripts/deploy_cloud_run.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID, e.g. PROJECT_ID=my-gcp-project}"
REGION="${REGION:-us-east1}"
SERVICE="${SERVICE:-hotel-cancel-api}"

if [[ ! -f models/model.joblib ]]; then
  echo "models/model.joblib not found — train first (python -m src.train)" >&2
  exit 1
fi

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# --source . uses our Dockerfile; .gcloudignore makes sure models/ is uploaded.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --port 8080

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo
echo "Deployed: $URL"
echo "Health:   curl $URL/health"
echo "Predict:  curl -X POST $URL/predict -H 'Content-Type: application/json' -d @scripts/sample_request.json"
