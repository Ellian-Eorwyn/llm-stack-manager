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

# shellcheck source=scripts/stack-services.sh
source "${STACK_DIR}/scripts/stack-services.sh"
ALL_SERVICES=("${STACK_STOPPABLE_SERVICES[@]}")

DEFAULT_CHAT_BACKEND="llm-a"
DEFAULT_CONFIG_MARKER="${STACK_DIR}/config/default-saved-config"
if [[ -f "${DEFAULT_CONFIG_MARKER}" ]]; then
    resolved_backend="$(python3 - "${DEFAULT_CONFIG_MARKER}" <<'PYDEFAULT'
import json
import re
import sys
from pathlib import Path

marker = Path(sys.argv[1])
stack_dir = marker.parent.parent
name = re.sub(r"[^\w-]", "_", marker.read_text().strip())
saved_path = stack_dir / "config" / "saved" / f"{name}.json"
if not saved_path.exists():
    raise SystemExit(0)

# The manager's own writer, not a copy of it. This block used to carry its own
# legacy map, its own CODE_* mirror table and its own quoting -- three tables
# that had to be kept in step with `web/config_fields.py` and were not. The map
# still pointed `CHAT_MODEL_27B_PATH` at `CHAT_DENSE_MODEL_PATH`, itself a legacy
# name two renames ago, so the one path that runs before anything else on a
# reboot was the one path reintroducing the spellings every other path had
# retired. It also wrote every key in the profile literally, with no allow-list.
# Both, and in this order, matching `scripts/lib/boot-services.py`: `config_env`
# reaches `core`, which imports `setup_engine` from `scripts/`.
sys.path.insert(0, str(stack_dir / "scripts"))
sys.path.insert(0, str(stack_dir / "web"))
import config_env  # noqa: E402

data = json.loads(saved_path.read_text())
updates = {k: str(v) for k, v in data.items() if not k.startswith("_") and isinstance(v, str)}
updates = config_env.filter_config_updates(updates)
updates = config_env.apply_code_chat_mirrors(updates)
if updates:
    config_env.update_env_values(updates)


# A saved profile used to record which of three mutually exclusive units served
# the primary slot. There is one unit per slot now, so any profile that had a
# chat backend active resolves to the same one -- including old profiles naming
# a retired variant, which would otherwise resolve to a unit that no longer
# exists and leave the stack with no chat backend at all.
active = data.get("_active_chat_model") if isinstance(data.get("_active_chat_model"), dict) else {}
if active.get("variant") or active.get("service"):
    print("llm-a")
PYDEFAULT
)"
    if [[ -n "${resolved_backend}" ]]; then
        DEFAULT_CHAT_BACKEND="${resolved_backend}"
        echo "Using saved default config: $(cat "${DEFAULT_CONFIG_MARKER}") (${DEFAULT_CHAT_BACKEND})"
    fi
fi

# What to start, and on whose authority. `scripts/lib/boot-services.py` has the
# rules; the short version is that config/service-expectations.json wins over
# the component selection, and a unit an operator switched off is never started.
#
# This used to be an if/else on LLM_STACK_SELECTED_COMPONENTS where the else
# branch started *everything* -- and llm-stack-restore.service sets no such
# variable, so every boot took it. On this host that meant starting
# llm-b, a second 27B, onto a GPU already holding the primary;
# service-expectations.json had said `off` for months and nothing here read it.
#
# The component-to-unit map moved to setup_engine.COMPONENT_SERVICES rather than
# staying spelled out below, which is the collapse tests/test_slot_registry.py
# describes as "needs Python on the install path".
# A read loop rather than `mapfile`: macOS ships bash 3.2, which has neither
# `mapfile` nor `readarray`, and this script runs on both platforms.
DEFAULT_SERVICES=()
while IFS= read -r boot_unit; do
    [[ -n "${boot_unit}" ]] && DEFAULT_SERVICES+=("${boot_unit}")
done < <(python3 "${STACK_DIR}/scripts/lib/boot-services.py" "${DEFAULT_CHAT_BACKEND}" || true)
if [[ ${#DEFAULT_SERVICES[@]} -eq 0 ]]; then
    echo "Refusing to continue: boot-services.py named no services to start." >&2
    echo "Nothing has been stopped. Check config/service-expectations.json and" >&2
    echo "config/install-state.json, or pass LLM_STACK_SELECTED_COMPONENTS." >&2
    exit 1
fi
echo "Services to start: ${DEFAULT_SERVICES[*]}"

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
echo ""
echo "Active default chat backend: ${DEFAULT_CHAT_BACKEND}"
