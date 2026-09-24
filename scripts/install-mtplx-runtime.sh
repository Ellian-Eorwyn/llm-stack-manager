#!/usr/bin/env bash
# =============================================================================
# install-mtplx-runtime.sh
# Creates the Python environment the mtplx engine runs in, and fetches a pack.
#
# Its own venv, not deps/mlx-runtime-venv: MTPLX pins mlx 0.32, mlx-lm 0.31 and
# transformers < 5.15, and mlx-audio's tree has already moved past the last of
# those. One shared environment would let either upgrade break the other.
#
# The package and the pack are both pinned. A pack is a quantization recipe as
# well as weights, and its runtime contract (mtplx_runtime.json) is checked
# against the MTPLX version that loads it, so the two move together or not at
# all. Override with MTPLX_SPEC and --pack REPO@REVISION.
#
# Not run as root, for the reason install-mlx-runtime.sh gives: the venv has to
# belong to the user the LaunchAgents run as.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/scripts/cross-platform.sh"

if ! is_mac || [[ "$(uname -m)" != "arm64" ]]; then
    echo "MTPLX is Apple silicon only; on this host the chat slots stay on llama.cpp." >&2
    exit 1
fi
if [[ "${EUID}" -eq 0 ]]; then
    echo "Do not run this with sudo: the venv must belong to the user the" >&2
    echo "LaunchAgents run as. Run it as that user." >&2
    exit 1
fi

CONFIG_FILE="${STACK_DIR}/config/llm-stack.env"
[[ -f "${CONFIG_FILE}" ]] && { set -a; source "${CONFIG_FILE}"; set +a; }

VENV_DIR="${MTPLX_VENV:-${STACK_DIR}/deps/mtplx-venv}"
MTPLX_SPEC="${MTPLX_SPEC:-mtplx==2.12.0}"
# Qwen 3.8 27B Optimized Speed: 5.8 bits/weight, 19 GiB, 25 GiB peak. The 8-bit
# pack, 9-16% slower and near-lossless (docs/mtplx.md), is
#   --pack Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality@300a4ac6c6058585e80571ff6c910819711843ec
PACK="Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed@1d5087d2062c02b279180a53e4016cf9cd7a3d7e"
FETCH_PACK=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pack)    PACK="${2:?--pack needs REPO@REVISION}"; shift ;;
        --no-pack) FETCH_PACK=0 ;;
        --help|-h)
            echo "Usage: $0 [--pack REPO@REVISION] [--no-pack]"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

PYTHON="${MTPLX_PYTHON:-python3}"
command -v "${PYTHON}" >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }

echo "[mtplx] venv: ${VENV_DIR}"
mkdir -p "$(dirname "${VENV_DIR}")"
[[ -x "${VENV_DIR}/bin/python" ]] || "${PYTHON}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
echo "[mtplx] installing: ${MTPLX_SPEC}"
"${VENV_DIR}/bin/python" -m pip install --quiet "${MTPLX_SPEC}"

if (( FETCH_PACK )); then
    repo="${PACK%@*}"
    revision="${PACK##*@}"
    [[ "${revision}" == "${PACK}" ]] && revision="main"
    target="${STACK_DIR}/models/mlx/${repo##*/}"
    if [[ -f "${target}/mtplx_runtime.json" ]]; then
        echo "[mtplx] pack present at ${target#"${STACK_DIR}"/}"
    else
        echo "[mtplx] fetching ${repo}@${revision}"
        "${VENV_DIR}/bin/hf" download "${repo}" --revision "${revision}" --local-dir "${target}"
    fi
    echo "[mtplx] to serve it from LLM A, set in config/llm-stack.env:"
    echo "          LLM_A_ENGINE=mtplx"
    echo "          LLM_A_MODEL_PATH=${target}"
fi

echo "[mtplx] runtime ready: ${VENV_DIR}"
