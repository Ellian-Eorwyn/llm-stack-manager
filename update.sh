#!/usr/bin/env bash
# Update this stack from GitHub and rebuild external dependencies.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${STACK_DIR}/config/llm-stack.env"
REMOTE="${LLM_STACK_UPDATE_REMOTE:-origin}"
BRANCH="${LLM_STACK_UPDATE_BRANCH:-main}"
CHANNEL="${LLM_STACK_UPDATE_CHANNEL:-release}"
SKIP_DEPS=0
SKIP_INSTALL=0
SKIP_RESTART=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Updates this checkout from GitHub, then updates dependencies and systemd units.

Options:
  --release          Update to the latest GitHub release/tag (default)
  --branch [name]    Update from a branch instead of release tags (default: main)
  --remote [name]    Git remote to fetch from (default: origin)
  --skip-deps        Do not rebuild/update external dependencies
  --skip-install     Do not run install.sh even when root
  --skip-restart     Do not restart active stack services
  -h, --help         Show this help

Environment:
  LLM_STACK_UPDATE_CHANNEL=release|branch
  LLM_STACK_UPDATE_BRANCH=main
  LLM_STACK_UPDATE_REMOTE=origin
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            CHANNEL="release"
            shift
            ;;
        --branch)
            CHANNEL="branch"
            if [[ $# -gt 1 && "$2" != --* ]]; then
                BRANCH="$2"
                shift 2
            else
                shift
            fi
            ;;
        --remote)
            REMOTE="${2:-}"
            if [[ -z "${REMOTE}" ]]; then
                echo "--remote requires a value" >&2
                exit 2
            fi
            shift 2
            ;;
        --skip-deps)
            SKIP_DEPS=1
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        --skip-restart)
            SKIP_RESTART=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
fi

cd "${STACK_DIR}"
if [[ ! -d .git ]]; then
    echo "This directory is not a git repository yet: ${STACK_DIR}" >&2
    exit 1
fi

if ! git remote get-url "${REMOTE}" >/dev/null 2>&1; then
    echo "Missing git remote '${REMOTE}'. Add one first, for example:" >&2
    echo "  git remote add ${REMOTE} https://github.com/Ellian-Eorwyn/llm-stack-manager.git" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to update with uncommitted changes in ${STACK_DIR}." >&2
    git status --short >&2
    exit 1
fi

remote_repo_slug() {
    local url="$1"
    case "${url}" in
        https://github.com/*)
            url="${url#https://github.com/}"
            url="${url%.git}"
            ;;
        git@github.com:*)
            url="${url#git@github.com:}"
            url="${url%.git}"
            ;;
        *)
            return 1
            ;;
    esac
    [[ "${url}" == */* ]] || return 1
    printf '%s\n' "${url}"
}

latest_release_tag() {
    local repo_slug="$1"
    if command -v gh >/dev/null 2>&1 && [[ -n "${repo_slug}" ]]; then
        gh release view --repo "${repo_slug}" --json tagName --jq .tagName 2>/dev/null || true
        return
    fi
    git tag --sort=-version:refname | head -n 1
}

REMOTE_URL="$(git remote get-url "${REMOTE}")"
REPO_SLUG="$(remote_repo_slug "${REMOTE_URL}" || true)"
CURRENT_BRANCH="$(git branch --show-current || true)"

echo "Fetching ${REMOTE}..."
git fetch --tags --prune "${REMOTE}"

case "${CHANNEL}" in
    release)
        TARGET_TAG="$(latest_release_tag "${REPO_SLUG}")"
        if [[ -n "${TARGET_TAG}" ]]; then
            echo "Updating to latest release/tag: ${TARGET_TAG}"
            if [[ -n "${CURRENT_BRANCH}" ]]; then
                git merge --ff-only "${TARGET_TAG}"
            else
                git checkout --detach "${TARGET_TAG}"
            fi
        else
            echo "No GitHub release or tag found; falling back to ${REMOTE}/${BRANCH}."
            if [[ -n "${CURRENT_BRANCH}" ]]; then
                git merge --ff-only "${REMOTE}/${BRANCH}"
            else
                git checkout --detach "${REMOTE}/${BRANCH}"
            fi
        fi
        ;;
    branch)
        echo "Updating from branch: ${REMOTE}/${BRANCH}"
        if [[ -n "${CURRENT_BRANCH}" ]]; then
            git pull --ff-only "${REMOTE}" "${BRANCH}"
        else
            git checkout --detach "${REMOTE}/${BRANCH}"
        fi
        ;;
    *)
        echo "Invalid update channel: ${CHANNEL}. Expected release or branch." >&2
        exit 2
        ;;
esac

if [[ "${SKIP_DEPS}" != "1" ]]; then
    env HONCHO_ENABLED="${HONCHO_ENABLED:-off}" "${STACK_DIR}/scripts/install-dependencies.py" --update
else
    echo "Skipping dependency update."
fi

if [[ "${EUID}" -eq 0 && "${SKIP_INSTALL}" != "1" ]]; then
    bash "${STACK_DIR}/install.sh"
    if [[ "${SKIP_RESTART}" != "1" ]]; then
        source "${STACK_DIR}/scripts/cross-platform.sh"
        if is_linux; then
            mapfile -t active < <(systemctl list-units --type=service --state=active --no-legend 'chat-*.service' 'embed.service' 'rerank.service' 'task.service' 'ocr.service' 'glmocr-sdk.service' 'think.service' 'nothink.service' 'qwen-*' 'honcho-*.service' 'llm-manager.service' | awk '{print $1}' | sed 's/\.service$//')
        else
            active=(llm-manager chat-backend chat-backend-dense chat-backend-moe chat-backend-bee chat-proxy embed rerank task ocr glmocr-sdk honcho-api honcho-deriver think nothink)
        fi
        
        for svc in "${active[@]}"; do
            case "${svc}" in
                llm-manager|chat-backend|chat-backend-dense|chat-backend-moe|chat-backend-bee|chat-proxy|embed|rerank|task|ocr|glmocr-sdk|honcho-api|honcho-deriver|think|nothink)
                    if is_mac || svc_is_active "${svc}"; then
                        svc_restart "${svc}"
                    fi
                    ;;
            esac
        done
    else
        echo "Skipping service restarts."
    fi
else
    echo "Repo updated. Run sudo bash ${STACK_DIR}/install.sh to regenerate systemd units."
fi
