#!/usr/bin/env bash
# Run on a clean/self-hosted Ubuntu 24.04 NVIDIA test host after wizard installation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/llm-stack.env"
SELECTED=",${LLM_STACK_SELECTED_COMPONENTS:-},"
has() { [[ "${SELECTED}" == *",$1,"* ]]; }

has primary && curl -fsS "http://127.0.0.1:${CODE_PORT}/v1/models" | grep -q '"object"'
has primary && curl -fsS "http://127.0.0.1:${NOTHINK_PORT}/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"chat","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8}' | grep -q '"content"'
has embedding && curl -fsS "http://127.0.0.1:${EMBED_PORT}/v1/embeddings" -H 'Content-Type: application/json' -d '{"model":"embed","input":"test"}' | grep -q '"embedding"'
has task && curl -fsS "http://127.0.0.1:${TASK_PORT}/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"task","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8}' | grep -q '"content"'
if has ocr; then
  PIXEL='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
  curl -fsS "http://127.0.0.1:${LLM_MANAGER_PORT}/api/ocr/extract" -H 'Content-Type: application/json' -d "{\"image_base64\":\"${PIXEL}\",\"mime_type\":\"image/png\"}" | grep -q '"ok"'
fi
has glmocr-sdk && curl -fsS "http://127.0.0.1:${GLMOCR_SDK_PORT}/health" | grep -q '"status"'
has searxng && curl -fsS "http://127.0.0.1${SEARXNG_URL_PATH}/search?q=test&format=json" | grep -q '"results"'
has playwright && PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH}" node "${ROOT}/playwright/test-remote.js"
echo "NVIDIA integration checks passed"
