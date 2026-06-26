#!/usr/bin/env bash
# Update this git-friendly stack repo and rebuild external dependencies.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${STACK_DIR}/config/llm-stack.env"
if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
fi

cd "${STACK_DIR}"
if [[ ! -d .git ]]; then
    echo "This directory is not a git repository yet: ${STACK_DIR}" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to update with uncommitted changes in ${STACK_DIR}." >&2
    git status --short >&2
    exit 1
fi

git pull --ff-only
env HONCHO_ENABLED="${HONCHO_ENABLED:-off}" "${STACK_DIR}/scripts/install-dependencies.py" --update

if [[ "${EUID}" -eq 0 ]]; then
    bash "${STACK_DIR}/install.sh"
    mapfile -t active < <(systemctl list-units --type=service --state=active --no-legend 'chat-*.service' 'embed.service' 'rerank.service' 'task.service' 'ocr.service' 'glmocr-sdk.service' 'think.service' 'nothink.service' 'qwen-*' 'honcho-*.service' 'llm-manager.service' | awk '{print $1}' | sed 's/\.service$//')
    for svc in "${active[@]}"; do
        case "${svc}" in
            llm-manager|chat-backend|chat-backend-dense|chat-backend-moe|chat-backend-bee|chat-proxy|embed|rerank|task|ocr|glmocr-sdk|honcho-api|honcho-deriver|think|nothink)
                systemctl restart "${svc}"
                ;;
        esac
    done
else
    echo "Repo and dependencies updated. Run sudo bash ${STACK_DIR}/install.sh to regenerate systemd units."
fi
