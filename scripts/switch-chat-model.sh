#!/usr/bin/env bash
# =============================================================================
# switch-chat-model.sh
# Stops the currently running chat backend and starts the other one.
# The proxy (chat-proxy) stays running throughout — no downtime on 8003/8004
# beyond the model load time.
#
# Usage:  sudo bash switch-chat-model.sh dense
#         sudo bash switch-chat-model.sh moe
#         sudo bash switch-chat-model.sh bee
# =============================================================================
set -euo pipefail

VARIANT="${1:-}"

if [[ "${VARIANT}" != "dense" && "${VARIANT}" != "moe" && "${VARIANT}" != "bee" ]]; then
    echo "Usage: sudo bash switch-chat-model.sh [dense|moe|bee]"
    echo ""
    echo "Current status:"
    systemctl is-active chat-backend-dense 2>/dev/null && echo "  chat-backend-dense: active" || echo "  chat-backend-dense: inactive"
    systemctl is-active chat-backend-moe 2>/dev/null && echo "  chat-backend-moe: active" || echo "  chat-backend-moe: inactive"
    systemctl is-active chat-backend-bee 2>/dev/null && echo "  chat-backend-bee: active" || echo "  chat-backend-bee: inactive"
    systemctl is-active chat-proxy       2>/dev/null && echo "  chat-proxy:       active" || echo "  chat-proxy:       inactive"
    exit 1
fi

case "${VARIANT}" in
    dense) START_UNIT="chat-backend-dense" ;;
    moe)   START_UNIT="chat-backend-moe" ;;
    bee)   START_UNIT="chat-backend-bee" ;;
esac

echo "=== Switching chat backend to ${VARIANT} ==="
echo ""

# Stop any backend that can own CHAT_BACKEND_PORT before starting the target.
CHAT_BACKENDS=(
    chat-backend
    chat-backend-dense
    chat-backend-moe
    chat-backend-bee
    qwen-chat-backend
    qwen-chat-backend-27b
    qwen-chat-backend-35b
)

echo "[1/2] Stopping other chat backends..."
for svc in "${CHAT_BACKENDS[@]}"; do
    [[ "${svc}" == "${START_UNIT}" ]] && continue
    if systemctl is-active --quiet "${svc}" 2>/dev/null; then
        echo "  stopping ${svc}..."
        systemctl stop "${svc}"
    fi
done

# Start the requested variant
echo "[2/2] Starting ${START_UNIT}..."
systemctl start "${START_UNIT}"
echo "  started."

echo ""
echo "=== Done. Chat endpoints are loading the ${VARIANT} model. ==="
echo ""
echo "The proxy on ports 8003/8004 will return 503 until the model finishes"
echo "loading. Watch progress with:"
echo "  journalctl -u ${START_UNIT} -f"
