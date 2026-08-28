#!/usr/bin/env bash
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "${STACK_DIR}/config/llm-stack.env"
set +a

VENV_DIR="${MLX_RUNTIME_VENV:-${STACK_DIR}/deps/mlx-runtime-venv}"
export HF_HOME="${MLX_HF_HOME:-${STACK_DIR}/models/.cache/huggingface}"
export PYTHONUNBUFFERED=1

exec "${VENV_DIR}/bin/python" "${STACK_DIR}/scripts/mlx-parakeet-server.py" \
    --host "${TRANSCRIPT_HOST:-127.0.0.1}" \
    --port "${TRANSCRIPT_PORT:-8014}"
