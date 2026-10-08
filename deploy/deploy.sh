#!/bin/bash
# Azure deployment sequence for the passport verification pipeline.
# Run from the repository root (Passport_verification/).
# Needs 4 vCPU / 8 GiB: TruFor is OOM-killed at 2 vCPU / 4 GiB.
set -euo pipefail

RESOURCE_GROUP="rg-passport-verify-dewa"
LOCATION="centralindia"
ACR_NAME="passportverifyacr"
ENV_NAME="passport-verify-env"
APP_NAME="passport-verify-api"
IMAGE_NAME="passport-verify"
IMAGE_TAG="v2"

echo "=== 1. Resource group ==="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "=== 2. Container Registry ==="
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic

echo "=== 3. Build image in the cloud (no local push needed) ==="
az acr build --registry "$ACR_NAME" --image "${IMAGE_NAME}:${IMAGE_TAG}" --file deploy/Dockerfile .

echo "=== 4. Container Apps environment ==="
az containerapp env create --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" --location "$LOCATION"

echo "=== 5. Generate an API key and create the Container App ==="
API_KEY=$(openssl rand -hex 24)
echo "Generated API key (save this — it will not be shown again): $API_KEY"

az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENV_NAME" \
  --image "${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}" \
  --registry-server "${ACR_NAME}.azurecr.io" \
  --cpu 4 --memory 8Gi \
  --min-replicas 0 --max-replicas 2 \
  --secrets api-key="$API_KEY" \
  --env-vars API_KEY=secretref:api-key \
  --ingress external --target-port 8000

echo "=== Done ==="
az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv
