#!/usr/bin/env bash
# =============================================================================
# install-launchd-service.sh <service>
# Generate the LaunchAgent for one model service, on demand.
#
# The setup wizard installs only the components it was asked for, so on a Mac
# a service left out at setup has no plist -- and pressing Start in the manager
# then asked launchctl to bootstrap a file that does not exist. The launchers
# themselves work; there was simply nothing for launchd to run. This writes the
# same plist and wrapper `install.sh` would, for one service, so Start can.
#
# User domain only: a system-domain install needs root to write
# /Library/LaunchDaemons, and that is `install.sh`'s job.
#
# The name -> launcher table must match `install_mac_service` in install.sh;
# tests/test_validate_script.py holds the two together.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/scripts/cross-platform.sh"
_stack_dir="${STACK_DIR}"
set -a
# shellcheck disable=SC1091
source "${STACK_DIR}/config/llm-stack.env"
set +a
# The config names the installed tree; the plist should point at this one.
STACK_DIR="${_stack_dir}"

name="${1:?usage: install-launchd-service.sh <service>}"

if ! is_mac; then
    echo "install-launchd-service.sh is for macOS; use install.sh on Linux." >&2
    exit 2
fi
if [[ "${LLM_LAUNCHD_DOMAIN:-user}" == "system" ]]; then
    echo "${name} is not installed, and system-domain services need root: run install.sh." >&2
    exit 2
fi

case "${name}" in
    llm-a)        script="start-llm-a.sh";        description="LLM A - llama-server" ;;
    llm-a-proxy)  script="start-llm-a-proxy.sh";  description="LLM A Proxy - think/chat/code ports" ;;
    llm-b)        script="start-llm-b.sh";        description="LLM B - llama-server" ;;
    llm-b-proxy)  script="start-llm-b-proxy.sh";  description="LLM B Proxy - think/chat/code ports" ;;
    embed)
        case "${EMBED_ENGINE:-llamacpp}" in
            mlx) script="start-embed-mlx.sh" ;;
            *)   script="start-embed.sh" ;;
        esac
        description="LLM Embedding Model - ${EMBED_ENGINE:-llamacpp}" ;;
    rerank)       script="start-rerank.sh";       description="LLM Reranker Model - llama-server" ;;
    task)         script="start-task.sh";         description="LLM Task Model - llama-server" ;;
    ocr)          script="start-ocr.sh";          description="LLM OCR GLM-OCR Backend - llama-server" ;;
    llama-router)
        if [[ "${MODEL_ROUTER_ENABLED:-off}" != "on" ]]; then
            echo "llama-router runs only with MODEL_ROUTER_ENABLED=on." >&2
            exit 2
        fi
        script="start-model-router.sh"; description="LLM Model Router - on-demand auxiliary models" ;;
    *)
        # glmocr-sdk and the transcription sidecar need runtimes install.sh sets
        # up; the manager has its own fallback for the sidecar.
        echo "${name} cannot be installed on demand; run install.sh." >&2
        exit 2 ;;
esac

if [[ ! -x "${STACK_DIR}/scripts/${script}" && ! -f "${STACK_DIR}/scripts/${script}" ]]; then
    echo "Launcher ${script} is missing from ${STACK_DIR}/scripts." >&2
    exit 1
fi

SERVICE_USER="$(cp_stat_user "${STACK_DIR}")"
SERVICE_GROUP="$(cp_stat_group "${STACK_DIR}")"
mkdir -p "${STACK_DIR}/logs"
generate_launchd_plist "${name}" "${description}" "${script}"
_launchd_reset
echo "Installed $(svc_plist_path "${name}")"
