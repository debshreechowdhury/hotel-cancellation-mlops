#!/usr/bin/env bash
# =============================================================================
# One-shot local setup for macOS / Linux. Run from the repo root:
#
#   ./scripts/setup_local.sh
#
# It asks once for your Supabase Project URL + secret key (saved to the
# git-ignored .env), uploads the clean data to Supabase Storage and trains from
# there. (Alternative: GCS_BUCKET=<bucket> ./scripts/setup_local.sh uses Google
# Cloud Storage instead; SKIP_CLOUD=1 keeps everything local.)
#
# What it does (safe to re-run; finished steps are skipped):
#   1. finds Python 3.11–3.13 (installs 3.12 with Homebrew if needed)
#   2. creates .venv and installs requirements-dev.txt
#   3. finds the Kaggle archive.zip / hotel_bookings.csv and unzips it
#   4. runs preprocessing
#   5. uploads the clean data to Supabase Storage (or GCS)
#   6. checks the W&B login (prompts for your API key once)
#   7. trains random_forest, logistic_regression, then the default model
#   8. smoke-tests the API locally with curl
#   9. prints the metrics and the links to send back
#
# Optional env vars: DATA_ZIP=/path/to/archive.zip  PYTHON=python3.12
#                    SKIP_TRAIN=1  SKIP_API_TEST=1
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
ok()   { printf "\033[1;32m✓ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m! %s\033[0m\n" "$*"; }
die()  { printf "\033[1;31m✗ %s\033[0m\n" "$*" >&2; exit 1; }

# --- 1. Python ---------------------------------------------------------------
step "1/9 Finding Python 3.11–3.13"
py_ok() { "$1" -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; }
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.12 python3.11 python3.13 python3 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
    if command -v "$c" >/dev/null 2>&1 && py_ok "$c"; then PY="$c"; break; fi
  done
fi
if [[ -z "$PY" ]]; then
  if command -v brew >/dev/null 2>&1; then
    warn "No Python 3.11–3.13 found; installing python@3.12 with Homebrew"
    brew install python@3.12
    PY="$(brew --prefix python@3.12)/bin/python3.12"
  else
    die "Need Python 3.11, 3.12 or 3.13. Install from https://www.python.org/downloads/ and re-run."
  fi
fi
py_ok "$PY" || die "$PY is not Python 3.11–3.13"
ok "Using $("$PY" --version) at $(command -v "$PY" || echo "$PY")"

# --- 2. venv + deps ------------------------------------------------------------
step "2/9 Virtual environment and dependencies"
if [[ ! -x .venv/bin/python ]] || ! py_ok .venv/bin/python; then
  rm -rf .venv
  "$PY" -m venv ${VENV_ARGS:-} .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [[ -z "${SKIP_INSTALL:-}" ]]; then
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements-dev.txt
fi
ok "Dependencies installed in .venv"

# --- 3. raw data ---------------------------------------------------------------
step "3/9 Kaggle data"
RAW=data/raw/hotel_bookings.csv
if [[ -f "$RAW" ]]; then
  ok "$RAW already present"
else
  CANDIDATES=("${DATA_ZIP:-}"
              "$HOME/Downloads/archive.zip"
              "$HOME/Downloads/hotel-booking-demand.zip"
              "$HOME/Documents/hotel-cancellation-data/archive.zip"
              "$HOME/Downloads/hotel_bookings.csv"
              "$HOME/Documents/hotel-cancellation-data/hotel_bookings.csv")
  FOUND=""
  for f in "${CANDIDATES[@]}"; do [[ -n "$f" && -f "$f" ]] && { FOUND="$f"; break; }; done
  [[ -n "$FOUND" ]] || die "Couldn't find the Kaggle download. Get it from
  https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
  then re-run with: DATA_ZIP=/path/to/archive.zip ./scripts/setup_local.sh"
  mkdir -p data/raw
  if [[ "$FOUND" == *.zip ]]; then
    unzip -o -q "$FOUND" -d data/raw/
    # some zips nest the csv in a folder
    [[ -f "$RAW" ]] || find data/raw -name hotel_bookings.csv -exec mv {} "$RAW" \;
  else
    cp "$FOUND" "$RAW"
  fi
  [[ -f "$RAW" ]] || die "Unzipped $FOUND but no hotel_bookings.csv inside"
  ok "Extracted $RAW from $FOUND"
fi

# --- 4. preprocess -------------------------------------------------------------
step "4/9 Preprocessing"
python -m src.preprocess
ok "data/processed/hotel_bookings_clean.csv written"

# --- 5. cloud storage -------------------------------------------------------------
TRAIN_ARGS=()
step "5/9 Cloud storage (Supabase)"
# Load saved credentials, if any
if [[ -f .env ]]; then set -a; # shellcheck disable=SC1091
  source .env; set +a; fi
if [[ -z "${GCS_BUCKET:-}" && ( -z "${SUPABASE_URL:-}" || -z "${SUPABASE_KEY:-}" ) && -z "${SKIP_CLOUD:-}" ]]; then
  echo "Supabase -> your project -> Project Settings:"
  echo "  * Data API  -> Project URL      (looks like https://abcd1234.supabase.co)"
  echo "  * API Keys  -> Secret key       (starts with sb_secret_ ; click the eye icon to reveal)"
  read -r -p "Paste Project URL: " SUPABASE_URL
  read -r -s -p "Paste Secret key (hidden): " SUPABASE_KEY; echo
  [[ "$SUPABASE_URL" == https://*.supabase.co* ]] || die "That doesn't look like a Supabase Project URL"
  [[ -n "$SUPABASE_KEY" ]] || die "Empty key"
  touch .env
  grep -v '^SUPABASE_URL=\|^SUPABASE_KEY=' .env > .env.tmp || true
  SUPABASE_URL="${SUPABASE_URL%/}"; SUPABASE_URL="${SUPABASE_URL%/rest/v1}"
  printf 'SUPABASE_URL=%s\nSUPABASE_KEY=%s\n' "$SUPABASE_URL" "$SUPABASE_KEY" >> .env.tmp
  mv .env.tmp .env && chmod 600 .env
  export SUPABASE_URL SUPABASE_KEY
  ok "Saved to .env (git-ignored)"
fi
if [[ -n "${SUPABASE_URL:-}" && -n "${SUPABASE_KEY:-}" ]]; then
  python -m src.ingest upload --set data.source=supabase
  TRAIN_ARGS=(--set data.source=supabase)
  ok "Training will download data from Supabase Storage"
elif [[ -n "${GCS_BUCKET:-}" ]]; then
  command -v gcloud >/dev/null 2>&1 || die "gcloud not found. Install: brew install --cask google-cloud-sdk"
  gcloud auth application-default print-access-token >/dev/null 2>&1 || gcloud auth application-default login
  gcloud storage buckets describe "gs://$GCS_BUCKET" >/dev/null 2>&1 \
    || gcloud storage buckets create "gs://$GCS_BUCKET" --location=us-east1
  export GCS_BUCKET
  python -m src.ingest upload --set data.source=gcs
  TRAIN_ARGS=(--set data.source=gcs)
  ok "Training will download data from gs://$GCS_BUCKET"
else
  warn "No cloud storage configured — training from the local file (SKIP_CLOUD set)."
  TRAIN_ARGS=(--set data.source=local)
fi

# --- 6. W&B ----------------------------------------------------------------------
step "6/9 Weights & Biases login"
if ! python -c "import wandb, sys; sys.exit(0 if wandb.Api().api_key else 1)" >/dev/null 2>&1; then
  warn "Paste your W&B API key from https://wandb.ai/authorize when prompted"
  wandb login
fi
ENTITY=$(python -c "import wandb; print(wandb.Api().default_entity or '')" 2>/dev/null || true)
ok "W&B logged in${ENTITY:+ as $ENTITY}"

# --- 7. train ----------------------------------------------------------------------
if [[ -z "${SKIP_TRAIN:-}" ]]; then
  step "7/9 Training 3 models (the default model trains last and is the one deployed)"
  python -m src.train ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} --set model.type=random_forest
  python -m src.train ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} --set model.type=logistic_regression
  python -m src.train ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}
  ok "Model saved to models/model.joblib"
else
  warn "7/9 SKIP_TRAIN set — skipping training"
fi

# --- 8. API smoke test ---------------------------------------------------------------
if [[ -z "${SKIP_API_TEST:-}" && -f models/model.joblib ]]; then
  step "8/9 Testing the API locally"
  uvicorn api.main:app --port 8080 >/tmp/hotel_api.log 2>&1 &
  API_PID=$!
  trap 'kill $API_PID 2>/dev/null || true' EXIT
  for _ in $(seq 1 30); do curl -sf localhost:8080/health >/dev/null && break; sleep 1; done
  RESP=$(curl -sf -X POST localhost:8080/predict -H "Content-Type: application/json" \
         -d @scripts/sample_request.json) || { cat /tmp/hotel_api.log; die "API test failed"; }
  echo "$RESP"
  kill $API_PID 2>/dev/null || true
  ok "API returns predictions"
fi

# --- 9. summary --------------------------------------------------------------------------
step "9/9 Summary — copy everything below this line and send it back"
python - <<'EOF'
import json, pathlib
p = pathlib.Path("models/metadata.json")
if not p.exists():
    print("no models/metadata.json yet")
else:
    m = json.loads(p.read_text())
    print("model:", m["model_type"], m["model_version"])
    print("wandb run:", m.get("wandb_run_url"))
    for k, v in m["metrics"].items():
        print(f"  {k:<16} {v:.4f}")
    print("top features:", ", ".join(f["feature"] for f in m["top_features"][:5]))
EOF
echo "W&B project: https://wandb.ai/${ENTITY:-<entity>}/hotel-cancellation"
echo
echo "Next: deploy the API ->  ./scripts/deploy_azure.sh"
echo "      run the app    ->  API_URL=http://localhost:8080 streamlit run frontend/app.py"
