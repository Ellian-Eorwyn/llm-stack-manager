#!/usr/bin/env bash
# =============================================================================
# install-mlx-runtime.sh
# Creates the Python environment the MLX services run in, and fetches their
# models.
#
# Separate from the manager's own venv on purpose. The manager deliberately
# depends on nothing but Flask (web/requirements.txt), and the MLX runtime is
# the opposite: mlx, mlx-embeddings, mlx-audio, fastapi, uvicorn, numpy and
# their transitive tree. Keeping them apart means an MLX upgrade cannot take
# the management UI down with it, and the UI stays installable on a host that
# is not serving any models at all.
#
# Not run as root: Homebrew and the per-user LaunchAgents these serve both
# want the service user, and a root-owned venv would be unusable by them.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/scripts/cross-platform.sh"

if ! is_mac; then
    echo "The MLX runtime is Apple silicon only." >&2
    exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "MLX requires Apple silicon (arm64)." >&2
    exit 1
fi
if [[ "${EUID}" -eq 0 ]]; then
    echo "Do not run this with sudo: the venv must belong to the user the" >&2
    echo "LaunchAgents run as. Run it as that user." >&2
    exit 1
fi

CONFIG_FILE="${STACK_DIR}/config/llm-stack.env"
[[ -f "${CONFIG_FILE}" ]] && { set -a; source "${CONFIG_FILE}"; set +a; }

VENV_DIR="${MLX_RUNTIME_VENV:-${STACK_DIR}/deps/mlx-runtime-venv}"
export HF_HOME="${MLX_HF_HOME:-${STACK_DIR}/models/.cache/huggingface}"
MODEL_DIR="${STACK_DIR}/models/mlx"

WITH_EMBED=1
WITH_TRANSCRIBE=1
FETCH_MODELS=1
for arg in "$@"; do
    case "${arg}" in
        --embed-only)      WITH_TRANSCRIBE=0 ;;
        --transcribe-only) WITH_EMBED=0 ;;
        --no-models)       FETCH_MODELS=0 ;;
        --help|-h)
            echo "Usage: $0 [--embed-only|--transcribe-only] [--no-models]"
            exit 0 ;;
        *) echo "Unknown option: ${arg}" >&2; exit 1 ;;
    esac
done

PYTHON="${MLX_RUNTIME_PYTHON:-python3}"
command -v "${PYTHON}" >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }

echo "[mlx] venv: ${VENV_DIR}"
mkdir -p "$(dirname "${VENV_DIR}")" "${MODEL_DIR}" "${HF_HOME}"
[[ -x "${VENV_DIR}/bin/python" ]] || "${PYTHON}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip

# python-multipart: FastAPI will not accept a form upload without it, and the
# transcription endpoint takes one. Once pulled in by an older dependency tree,
# its absence stopped the Parakeet server at startup.
PACKAGES=(fastapi uvicorn numpy huggingface_hub python-multipart)
(( WITH_EMBED ))      && PACKAGES+=(mlx mlx-embeddings)
# Nemotron 3 Diarization (mlx_audio.vad.models.nemotron_diarization) was merged
# after mlx-audio 0.5.5, the newest release. Pinned to the commit rather than
# to main; return to a plain `mlx-audio>=<next release>` once one ships it.
MLX_AUDIO_SPEC="${MLX_AUDIO_SPEC:-mlx-audio @ git+https://github.com/Blaizzy/mlx-audio@9ada37c1e33cfc99a7bdde0a902c0d4a0b913183}"
(( WITH_TRANSCRIBE )) && PACKAGES+=(mlx "${MLX_AUDIO_SPEC}")

echo "[mlx] installing: ${PACKAGES[*]}"
"${VENV_DIR}/bin/python" -m pip install --quiet "${PACKAGES[@]}"

# The pinned commit still calls itself 0.5.5, so pip counts an existing 0.5.5
# from PyPI as satisfying it and installs nothing, which leaves an upgraded
# Mac without the diarization model. Compare the commit pip recorded instead.
if (( WITH_TRANSCRIBE )) && [[ "${MLX_AUDIO_SPEC}" == *"@ git+"* ]]; then
    wanted_commit="${MLX_AUDIO_SPEC##*@}"
    installed_commit="$("${VENV_DIR}/bin/python" - <<'PY'
import json
from importlib.metadata import distribution
try:
    info = json.loads(distribution("mlx-audio").read_text("direct_url.json") or "{}")
    print(info.get("vcs_info", {}).get("commit_id", ""))
except Exception:
    print("")
PY
)"
    if [[ "${installed_commit}" != "${wanted_commit}" ]]; then
        echo "[mlx] mlx-audio: replacing ${installed_commit:-a PyPI release} with ${wanted_commit}"
        "${VENV_DIR}/bin/python" -m pip install --quiet --force-reinstall --no-deps "${MLX_AUDIO_SPEC}"
    fi
fi

if (( FETCH_MODELS )); then
    # Pinned to the revisions in config/mlx-models.lock.json rather than to a
    # branch: an embedding model that silently changes revision changes every
    # vector it has ever produced, and nothing downstream would notice.
    STACK_DIR="${STACK_DIR}" WITH_EMBED="${WITH_EMBED}" WITH_TRANSCRIBE="${WITH_TRANSCRIBE}" \
        "${VENV_DIR}/bin/python" - <<'PY'
import json
import os
import pathlib
import sys

stack = pathlib.Path(os.environ["STACK_DIR"])
lock_path = stack / "config" / "mlx-models.lock.json"
if not lock_path.exists():
    print(f"[mlx] no lock file at {lock_path}; skipping model download")
    sys.exit(0)

lock = json.loads(lock_path.read_text())
from huggingface_hub import snapshot_download

# --embed-only / --transcribe-only used to install one runtime and download
# both models anyway.
wanted = {"embedding": os.environ.get("WITH_EMBED") == "1",
          "transcription": os.environ.get("WITH_TRANSCRIBE") == "1",
          "diarization": os.environ.get("WITH_TRANSCRIBE") == "1"}
for name, entry in (lock.get("models") or {}).items():
    if not wanted.get(name, True):
        print(f"[mlx] {name}: not requested; skipping")
        continue
    repo, revision = entry.get("repository"), entry.get("revision")
    local = entry.get("local_path")
    if not (repo and local):
        continue
    target = stack / local
    if target.exists() and any(target.iterdir()):
        print(f"[mlx] {name}: present at {local}")
        continue
    print(f"[mlx] {name}: fetching {repo}@{revision or 'main'}")
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(target))
PY
fi

echo "[mlx] runtime ready: ${VENV_DIR}"
echo "[mlx] set MLX_EMBED_MODEL_PATH / MLX_PARAKEET_MODEL_PATH / MLX_DIARIZATION_MODEL_PATH in config/llm-stack.env"
