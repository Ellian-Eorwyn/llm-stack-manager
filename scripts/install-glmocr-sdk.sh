#!/usr/bin/env bash
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"
VENV_DIR="${GLMOCR_SDK_VENV_DIR:-${STACK_DIR}/deps/glmocr-sdk-venv}"
VERSION="${GLMOCR_SDK_VERSION:-0.1.5}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install "glmocr[selfhosted,server]==${VERSION}"
"${VENV_DIR}/bin/python" - <<'PY'
import flask
import glmocr
print("GLM-OCR SDK imports verified")
PY

if [[ "${GLMOCR_PRELOAD_LAYOUT_MODEL:-on}" == "on" ]]; then
  "${VENV_DIR}/bin/python" - "${GLMOCR_LAYOUT_MODEL_DIR:-PaddlePaddle/PP-DocLayoutV3_safetensors}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])
print("GLM-OCR layout model cached")
PY
fi
