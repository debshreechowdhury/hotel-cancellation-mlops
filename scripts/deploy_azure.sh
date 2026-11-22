#!/usr/bin/env bash
# =============================================================================
# Deploy the prediction API to Azure Functions (Flex Consumption, Python 3.11).
# Works with an "Azure for Students" subscription — no credit card needed.
#
#   ./scripts/deploy_azure.sh                  # first deploy and every redeploy
#   LOCATION=westus2 ./scripts/deploy_azure.sh # if your subscription blocks eastus
#
# Needs a trained model in models/ (run ./scripts/setup_local.sh first).
# Resource names are saved in .azure-deploy.env so redeploys reuse them.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
ok()   { printf "\033[1;32m✓ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m! %s\033[0m\n" "$*"; }
die()  { printf "\033[1;31m✗ %s\033[0m\n" "$*" >&2; exit 1; }

[[ -f models/model.joblib ]] || die "models/model.joblib not found — run ./scripts/setup_local.sh first"

# --- tools ---------------------------------------------------------------------
step "1/6 Checking tools (Azure CLI, Functions Core Tools)"
if ! command -v az >/dev/null 2>&1; then
  command -v brew >/dev/null 2>&1 || die "Install the Azure CLI: https://learn.microsoft.com/cli/azure/install-azure-cli"
  brew install azure-cli
fi
if ! command -v func >/dev/null 2>&1; then
  command -v brew >/dev/null 2>&1 || die "Install Azure Functions Core Tools v4: https://learn.microsoft.com/azure/azure-functions/functions-run-local"
  brew tap azure/functions
  brew install azure-functions-core-tools@4
fi
ok "az $(az version --query '"azure-cli"' -o tsv 2>/dev/null), func $(func --version)"

# --- login -----------------------------------------------------------------------
step "2/6 Azure login"
if ! az account show >/dev/null 2>&1; then
  az login >/dev/null
fi
SUB=$(az account show --query name -o tsv)
ok "Using subscription: $SUB"
for ns in Microsoft.Web Microsoft.Storage Microsoft.Insights Microsoft.OperationalInsights; do
  az provider register --namespace "$ns" >/dev/null 2>&1 || true
done

# --- names (persisted) -----------------------------------------------------------
STATE=.azure-deploy.env
if [[ -f "$STATE" ]]; then
  # shellcheck disable=SC1090
  source "$STATE"
else
  SUFFIX=$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 6)
  RG="hotel-cancel-rg"
  APP="hotel-cancel-api-$SUFFIX"
  STORAGE="hotelcancel$SUFFIX"
fi
LOCATION="${LOCATION:-${SAVED_LOCATION:-eastus}}"

# --- infrastructure ----------------------------------------------------------------
step "3/6 Creating resources in $LOCATION (skipped if they exist)"
region_hint() {
  warn "Azure for Students only allows some regions. Try another, e.g.:"
  warn "  LOCATION=westus2 ./scripts/deploy_azure.sh   (or centralus, eastus2, westeurope)"
  warn "Regions with Flex Consumption: az functionapp list-flexconsumption-locations -o table"
}
if [[ "$(az group exists -n "$RG")" != "true" ]]; then
  az group create -n "$RG" -l "$LOCATION" -o none || { region_hint; exit 1; }
fi
if ! az storage account show -n "$STORAGE" -g "$RG" >/dev/null 2>&1; then
  az storage account create -n "$STORAGE" -g "$RG" -l "$LOCATION" --sku Standard_LRS \
    --min-tls-version TLS1_2 --allow-blob-public-access false -o none || { region_hint; exit 1; }
fi
if ! az functionapp show -n "$APP" -g "$RG" >/dev/null 2>&1; then
  az functionapp create -n "$APP" -g "$RG" --storage-account "$STORAGE" \
    --flexconsumption-location "$LOCATION" --runtime python --runtime-version 3.11 -o none \
    || { region_hint; exit 1; }
fi
printf 'RG=%s\nAPP=%s\nSTORAGE=%s\nSAVED_LOCATION=%s\n' "$RG" "$APP" "$STORAGE" "$LOCATION" > "$STATE"
ok "Function app: $APP (resource group $RG)"

# --- package ---------------------------------------------------------------------------
step "4/6 Packaging API + model"
BUILD=build/azure
rm -rf "$BUILD" && mkdir -p "$BUILD/models"
cp deploy/azure/function_app.py deploy/azure/host.json deploy/azure/requirements.txt deploy/azure/.funcignore "$BUILD/"
cp -R api src "$BUILD/"
cp config.yaml "$BUILD/"
cp models/model.joblib "$BUILD/models/"
[[ -f models/metadata.json ]] && cp models/metadata.json "$BUILD/models/"
find "$BUILD" -name __pycache__ -prune -exec rm -rf {} +
ok "Package ready ($(du -sh "$BUILD" | cut -f1))"

# --- publish ---------------------------------------------------------------------------------
step "5/6 Publishing (remote build on Azure, ~3–5 min)"
(cd "$BUILD" && func azure functionapp publish "$APP" --python)

# --- verify ------------------------------------------------------------------------------------
step "6/6 Testing the live endpoint"
HOST=$(az functionapp show -n "$APP" -g "$RG" --query defaultHostName -o tsv)
URL="https://$HOST"
for i in $(seq 1 24); do
  if curl -sf "$URL/health" >/dev/null; then break; fi
  sleep 5
done
curl -s "$URL/health"; echo
curl -s -X POST "$URL/predict" -H "Content-Type: application/json" -d @scripts/sample_request.json; echo
echo
ok "Deployed: $URL"
echo "   Docs:    $URL/docs"
echo "   Predict: curl -X POST $URL/predict -H 'Content-Type: application/json' -d @scripts/sample_request.json"
echo
echo "Next: put  API_URL = \"$URL\"  in your Streamlit Cloud app secrets."
