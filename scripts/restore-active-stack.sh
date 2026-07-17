#!/usr/bin/env bash
# Starts the default core LLM stack: shared chat backend, proxy, embedding,
# reranker, and task model. The manager stays available throughout.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/cross-platform.sh
source "${STACK_DIR}/scripts/cross-platform.sh"

CONFIG_FILE="${STACK_DIR}/config/llm-stack.env"
if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
fi

ALL_SERVICES=(
    think
    nothink
    chat-backend
    chat-backend-dense
    chat-backend-moe
    chat-proxy
    chat-backend2
    chat-proxy2
    embed
    embed2
    rerank
    task
    ocr
    glmocr-sdk
    honcho-api
    honcho-deriver
    qwen-think
    qwen-nothink
    qwen-chat-backend
    qwen-chat-backend-27b
    qwen-chat-backend-35b
    qwen-chat-proxy
    qwen-embedding
    qwen-reranker
    qwen-task
)

DEFAULT_CHAT_BACKEND="chat-backend-dense"
DEFAULT_CONFIG_MARKER="${STACK_DIR}/config/default-saved-config"
if [[ -f "${DEFAULT_CONFIG_MARKER}" ]]; then
    resolved_backend="$(python3 - "${DEFAULT_CONFIG_MARKER}" <<'PYDEFAULT'
import json
import re
import sys
from pathlib import Path

marker = Path(sys.argv[1])
stack_dir = marker.parent.parent
config_path = stack_dir / "config" / "llm-stack.env"
name = re.sub(r"[^\w-]", "_", marker.read_text().strip())
saved_path = stack_dir / "config" / "saved" / f"{name}.json"
if not saved_path.exists():
    raise SystemExit(0)

data = json.loads(saved_path.read_text())
updates = {k: str(v) for k, v in data.items() if not k.startswith("_") and isinstance(v, str)}

legacy_map = {
    "CHAT_MODEL_27B_PATH": "CHAT_DENSE_MODEL_PATH",
    "CHAT_MMPROJ_27B_PATH": "CHAT_DENSE_MMPROJ_PATH",
    "CHAT_27B_CTX_SIZE": "CHAT_DENSE_CTX_SIZE",
    "CHAT_MODEL_35B_PATH": "CHAT_MOE_MODEL_PATH",
    "CHAT_MMPROJ_35B_PATH": "CHAT_MOE_MMPROJ_PATH",
    "CHAT_35B_CTX_SIZE": "CHAT_MOE_CTX_SIZE",
}
updates = {legacy_map.get(k, k): v for k, v in updates.items()}

code_to_chat = {
    "CODE_CTX_SIZE": ["CHAT_CTX_SIZE", "CHAT_DENSE_CTX_SIZE", "CHAT_MOE_CTX_SIZE"],
    "CODE_N_PARALLEL": ["CHAT_N_PARALLEL"],
    "CODE_THREADS": ["CHAT_THREADS"],
    "CODE_THREADS_BATCH": ["CHAT_THREADS_BATCH"],
    "CODE_N_GPU_LAYERS": ["CHAT_N_GPU_LAYERS"],
    "CODE_TENSOR_SPLIT": ["CHAT_TENSOR_SPLIT"],
    "CODE_SPLIT_MODE": ["CHAT_SPLIT_MODE"],
    "CODE_FLASH_ATTN": ["CHAT_FLASH_ATTN"],
    "CODE_CACHE_TYPE_K": ["CHAT_CACHE_TYPE_K"],
    "CODE_CACHE_TYPE_V": ["CHAT_CACHE_TYPE_V"],
    "CODE_BATCH_SIZE": ["CHAT_BATCH_SIZE"],
    "CODE_UBATCH_SIZE": ["CHAT_UBATCH_SIZE"],
    "CODE_NO_MMAP": ["CHAT_NO_MMAP"],
    "CODE_MLOCK": ["CHAT_MLOCK"],
    "CODE_GPU_VISIBLE_DEVICES": ["CHAT_GPU_VISIBLE_DEVICES"],
    "CODE_REASONING_FORMAT": ["CHAT_REASONING_FORMAT"],
    "CODE_FIT": ["CHAT_FIT"],
}
for code_key, chat_keys in code_to_chat.items():
    if code_key in updates:
        for chat_key in chat_keys:
            if chat_key not in updates:
                updates[chat_key] = updates[code_key]

content = config_path.read_text()

def quote_env(value: str) -> str:
    if value == "":
        return '""'
    if re.fullmatch(r"[A-Za-z0-9_./,:@%+-]+", value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

for key, value in updates.items():
    rendered = quote_env(value)
    pattern = re.compile(r"^" + re.escape(key) + r"=.*$", re.MULTILINE)
    if pattern.search(content):
        content = pattern.sub(f"{key}={rendered}", content, count=1)
    else:
        content += f"\n{key}={rendered}\n"
config_path.write_text(re.sub(r"\n{3,}", "\n\n", content))

active = data.get("_active_chat_model") if isinstance(data.get("_active_chat_model"), dict) else {}
variant = active.get("variant")
service = active.get("service")
if variant == "moe":
    print("chat-backend-moe")
elif variant == "dense":
    print("chat-backend-dense")
elif service == "chat-backend":
    print("chat-backend")
PYDEFAULT
)"
    if [[ -n "${resolved_backend}" ]]; then
        DEFAULT_CHAT_BACKEND="${resolved_backend}"
        echo "Using saved default config: $(cat "${DEFAULT_CONFIG_MARKER}") (${DEFAULT_CHAT_BACKEND})"
    fi
fi

DEFAULT_SERVICES=(
    "${DEFAULT_CHAT_BACKEND}"
    chat-proxy
    chat-backend2
    chat-proxy2
    embed
    embed2
    rerank
    task
)
if [[ "${HONCHO_ENABLED:-off}" == "on" ]]; then
    DEFAULT_SERVICES+=(honcho-api honcho-deriver)
fi

if ! svc_is_active llm-manager 2>/dev/null; then
    echo "  starting llm-manager..."
    svc_start llm-manager
fi

echo "=== LLM Stack: Default Core Mode ==="
echo ""
echo "[1/2] Stopping core LLM services..."
for svc in "${ALL_SERVICES[@]}"; do
    if svc_is_active "${svc}" 2>/dev/null; then
        echo "  stopping ${svc}..."
        svc_stop "${svc}"
    fi
done
echo "  done."

echo ""
echo "[2/2] Starting default core services..."
for svc in "${DEFAULT_SERVICES[@]}"; do
    echo "  starting ${svc}..."
    svc_start "${svc}"
done
echo "  done."

echo ""
echo "=== Default core mode active ==="
svc_status_all "${DEFAULT_SERVICES[@]}"
echo ""
echo "Endpoints:"
echo "  http://localhost:8003/v1  - thinking"
echo "  http://localhost:8004/v1  - chat"
echo "  http://localhost:8008/v1  - code"
echo "  http://localhost:8005/v1  - embeddings"
echo "  http://localhost:8006/v1  - reranker"
echo "  http://localhost:8007/v1  - task model"
echo "  http://localhost:8009/v1  - OCR (start ocr service when needed)"
if [[ "${HONCHO_ENABLED:-off}" == "on" ]]; then
    echo "  ${HONCHO_URL:-http://localhost:${HONCHO_PORT}}      - Honcho memory"
fi
echo ""
echo "Active default chat backend: ${DEFAULT_CHAT_BACKEND}"
