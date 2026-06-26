#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-your-gcp-project}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-liveops-policy-lab}"
BQ_DATASET="${BQ_DATASET:-liveops_policy_lab}"
RUNTIME_MODE="${RUNTIME_MODE:-demo}"
DATA_SOURCE="${DATA_SOURCE:-repo}"
USE_BIGQUERY="${USE_BIGQUERY:-false}"
ENABLE_GEMINI="${ENABLE_GEMINI:-false}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com bigquery.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com secretmanager.googleapis.com

ENV_VARS="RUNTIME_MODE=${RUNTIME_MODE},DATA_SOURCE=${DATA_SOURCE},USE_BIGQUERY=${USE_BIGQUERY},ENABLE_GEMINI=${ENABLE_GEMINI}"
if [[ "${USE_BIGQUERY}" == "true" ]]; then
  ENV_VARS="${ENV_VARS},GCP_PROJECT=${PROJECT_ID},BQ_DATASET=${BQ_DATASET}"
fi
if [[ -n "${GCS_BUCKET:-}" ]]; then
  ENV_VARS="${ENV_VARS},GCS_BUCKET=${GCS_BUCKET}"
fi

if [[ "${ENABLE_GEMINI}" == "true" && -n "${GEMINI_API_KEY:-}" ]]; then
  echo -n "$GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=- 2>/dev/null || \
    echo -n "$GEMINI_API_KEY" | gcloud secrets versions add gemini-api-key --data-file=-
  gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --allow-unauthenticated \
    --set-env-vars "$ENV_VARS" \
    --set-secrets GEMINI_API_KEY=gemini-api-key:latest
else
  gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --allow-unauthenticated \
    --set-env-vars "$ENV_VARS"
fi

