#!/usr/bin/env bash
# Update this stack from GitHub and rebuild external dependencies.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${STACK_DIR}/config/llm-stack.env"
REMOTE="${LLM_STACK_UPDATE_REMOTE:-origin}"
BRANCH="${LLM_STACK_UPDATE_BRANCH:-main}"
CHANNEL="${LLM_STACK_UPDATE_CHANNEL:-branch}"
SKIP_DEPS=0
SKIP_INSTALL=0
SKIP_RESTART=0
MANAGER_ONLY=0

# Restarting these costs nothing but a dropped in-flight request, so a
# manager-only update can always do it. Model backends are deliberately absent:
# restarting one reloads tens of GB of weights and discards its warm prompt
# cache, which is far too expensive to do on every code update.
# transcript-backend belongs here: it idle-unloads anyway, so a restart
# discards nothing a request would not have discarded a few minutes later.
CHEAP_RESTART_SERVICES=(llm-manager llm-a-proxy llm-b-proxy glmocr-sdk playwright-server transcript-backend)
# Changes under these paths mean a model backend really is running stale code.
# web/deploy.py holds the same list so the manager's drift badge and this
# post-update report agree about what costs a model reload; tests/test_deploy.py
# asserts the two have not drifted apart.
BACKEND_SENSITIVE_PATHS=(
    "scripts/start-chat-backend"
    "scripts/start-embed"
    "scripts/start-rerank"
    "scripts/start-task"
    "scripts/start-ocr"
    # The one launcher the consolidated slots run through.
    "scripts/start-backend.sh"
    # Sourced by every launcher, and it decides which flags reach llama-server.
    "scripts/lib/"
    # The launchers consult the budget model at startup to skip flags the
    # loaded model cannot act on, so a change here changes the next launch.
    "web/budget.py"
    # What a slot *is*: its flags, defaults and engine. A change here changes
    # the command line a backend is next started with, exactly as editing its
    # launcher used to.
    "web/backends/"
)

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Updates this checkout from GitHub, then updates dependencies and systemd units.

Options:
  --manager-only     Fast path: pull, regenerate units, restart only the manager
                     and proxies. Skips dependency builds (no llama.cpp rebuild)
                     and leaves model backends running with their warm caches.
  --release          Update to the latest GitHub release/tag
  --branch [name]    Update from a branch (the default; branch: main)
  --remote [name]    Git remote to fetch from (default: origin)
  --skip-deps        Do not rebuild/update external dependencies
  --skip-install     Do not run install.sh even when root
  --skip-restart     Do not restart active stack services
  -h, --help         Show this help

Environment:
  LLM_STACK_UPDATE_CHANNEL=release|branch   (default branch; read from the
                                            environment, not config/llm-stack.env)
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
        --manager-only)
            MANAGER_ONLY=1
            SKIP_DEPS=1
            CHANNEL="branch"
            shift
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

# Honour the same env gate install.sh uses, so setting it once covers both.
[[ "${LLM_STACK_SKIP_DEP_UPDATE:-0}" == "1" ]] && SKIP_DEPS=1

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
REVISION_BEFORE="$(git rev-parse HEAD)"

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

REVISION_AFTER="$(git rev-parse HEAD)"
if [[ "${REVISION_BEFORE}" == "${REVISION_AFTER}" ]]; then
    echo "Already up to date at ${REVISION_AFTER:0:8}."
else
    echo "Updated ${REVISION_BEFORE:0:8} -> ${REVISION_AFTER:0:8}."
fi

# Which model backends, if any, are now running stale launcher code. Reported
# rather than acted on: the user decides when to pay for a model reload.
changed_backend_files() {
    [[ "${REVISION_BEFORE}" == "${REVISION_AFTER}" ]] && return 0
    local patterns=()
    local path
    for path in "${BACKEND_SENSITIVE_PATHS[@]}"; do
        patterns+=("${path}*")
    done
    git diff --name-only "${REVISION_BEFORE}" "${REVISION_AFTER}" -- "${patterns[@]}" 2>/dev/null || true
}

if [[ "${SKIP_DEPS}" != "1" ]]; then
    "${STACK_DIR}/scripts/install-dependencies.py" --update
else
    echo "Skipping dependency update (no llama.cpp rebuild)."
fi

# svc_is_active / svc_restart for both platforms. Sourced from this script's own
# tree: the config may point STACK_DIR at a different checkout.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/cross-platform.sh"

# The restart hint for a backend running stale launcher code.
restart_hint() {
    if is_mac && [[ "${LLM_LAUNCHD_DOMAIN:-user}" != "system" ]]; then
        echo "  launchctl kickstart -k gui/$(id -u)/com.llmstack.$1"
    elif is_mac; then
        echo "  sudo launchctl kickstart -k system/com.llmstack.$1"
    else
        echo "  sudo systemctl restart $1"
    fi
}

report_stale_backends() {
    local stale
    stale="$(changed_backend_files)"
    [[ -n "${stale}" ]] || return 0
    echo ""
    echo "Model backend launchers changed in this update:"
    printf '  %s\n' ${stale}
    echo "Those backends are still running the previous code. Restart one when"
    echo "you can afford the model reload, for example:"
    restart_hint llm-a
}

if is_mac && [[ "${LLM_LAUNCHD_DOMAIN:-user}" != "system" && "${SKIP_INSTALL}" != "1" ]]; then
    # --- macOS, user domain ---------------------------------------------------
    # No root is needed here, and asking for it would put root-owned plists in
    # ~/Library/LaunchAgents that launchctl refuses to load. This path used to
    # sit behind the root check below, so an update on a Mac pulled the code
    # and restarted nothing -- the manager went on serving the old version.
    #
    # Not install.sh: with no component selection it installs an agent for
    # every component, and launchd starts any KeepAlive agent it finds at the
    # next login. Refresh the agents that are installed and nothing else.
    shopt -s nullglob
    for plist in "$(svc_plist_dir)"/com.llmstack.*.plist; do
        svc="$(basename "${plist}" .plist)"
        svc="${svc#com.llmstack.}"
        bash "${STACK_DIR}/scripts/install-launchd-service.sh" "${svc}" >/dev/null 2>&1 \
            || echo "  ${svc}: agent left as it was (install.sh regenerates it)"
    done
    shopt -u nullglob

    if [[ "${SKIP_RESTART}" != "1" ]]; then
        source "${STACK_DIR}/scripts/stack-services.sh"
        if [[ "${MANAGER_ONLY}" == "1" ]]; then
            restart=("${CHEAP_RESTART_SERVICES[@]}")
        else
            restart=("${STACK_UPDATE_RESTART_SERVICES[@]}")
        fi
        # Running services only: on launchd, starting a job means loading it,
        # so restarting a stopped one would start it.
        for svc in "${restart[@]}"; do
            [[ "${svc}" == "llm-manager" ]] && continue
            if svc_is_active "${svc}"; then
                echo "Restarting ${svc}..."
                launchctl kickstart -k "$(svc_domain)/$(svc_label "${svc}")"
            fi
        done
        [[ "${MANAGER_ONLY}" == "1" ]] && report_stale_backends
        # Last, and by kickstart. The manager's Update button runs this script
        # as the manager's child, so stopping the manager stops this script
        # too: a bootout-then-bootstrap would never reach the bootstrap and the
        # manager would stay down. kickstart -k is one request to launchd, which
        # restarts the job whatever becomes of the process that asked.
        if svc_is_active llm-manager; then
            echo "Restarting llm-manager..."
            launchctl kickstart -k "$(svc_domain)/$(svc_label llm-manager)"
        fi
    else
        echo "Skipping service restarts."
    fi
elif [[ "${EUID}" -eq 0 && "${SKIP_INSTALL}" != "1" ]]; then
    # install.sh runs its own install-dependencies.py --update, which cmake-builds
    # llama.cpp. Skipping our call is not enough; the gate has to be handed down
    # or --skip-deps silently still pays for a CUDA rebuild. A manager-only
    # update also has no reason to reinstall SearXNG and Playwright.
    env LLM_STACK_SKIP_DEP_UPDATE="${SKIP_DEPS}" \
        LLM_STACK_SKIP_EXTERNAL_INSTALL="${MANAGER_ONLY}" \
        bash "${STACK_DIR}/install.sh"
    if [[ "${SKIP_RESTART}" != "1" && "${MANAGER_ONLY}" == "1" ]]; then
        # `is_mac ||` used to stand in front of this check, because on a Mac
        # svc_is_active read every service as inactive. It reads launchd
        # correctly now, and a stopped service stays stopped.
        for svc in "${CHEAP_RESTART_SERVICES[@]}"; do
            if svc_is_active "${svc}"; then
                echo "Restarting ${svc}..."
                svc_restart "${svc}"
            fi
        done
        report_stale_backends
    elif [[ "${SKIP_RESTART}" != "1" ]]; then
        # shellcheck source=scripts/stack-services.sh
        source "${STACK_DIR}/scripts/stack-services.sh"
        if is_linux; then
            mapfile -t active < <(systemctl list-units --type=service --state=active --no-legend 'chat-*.service' 'embed.service' 'rerank.service' 'task.service' 'ocr.service' 'glmocr-sdk.service' 'transcript-backend.service' 'playwright-server.service' 'think.service' 'nothink.service' 'qwen-*' 'llm-manager.service' | awk '{print $1}' | sed 's/\.service$//')
        else
            active=("${STACK_UPDATE_RESTART_SERVICES[@]}")
        fi

        for svc in "${active[@]}"; do
            if stack_contains "${svc}" "${STACK_UPDATE_RESTART_SERVICES[@]}"; then
                if svc_is_active "${svc}"; then
                    svc_restart "${svc}"
                fi
            fi
        done
    else
        echo "Skipping service restarts."
    fi
else
    echo "Repo updated. Run sudo bash ${STACK_DIR}/install.sh to regenerate systemd units."
fi
