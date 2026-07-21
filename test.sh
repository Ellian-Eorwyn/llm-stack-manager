#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.test-venv"
python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --quiet -r "${ROOT}/web/requirements.txt"
"${VENV}/bin/python" -m unittest discover -s "${ROOT}/tests" -v
node --test "${ROOT}/tests/test_cache_aware_scheduling.js"
bash -n "${ROOT}/install.sh" "${ROOT}/update.sh" "${ROOT}/validate.sh" "${ROOT}/bootstrap-ubuntu.sh" "${ROOT}/scripts/"*.sh
