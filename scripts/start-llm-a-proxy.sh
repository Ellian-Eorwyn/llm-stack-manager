#!/usr/bin/env bash
# =============================================================================
# start-llm-a-proxy.sh
# Shim. This proxy is served by scripts/start-proxy.sh, which builds its
# environment from web/backends/proxies.py.
#
# Kept as a name rather than deleted for the reason the backend shims are: a
# generated systemd unit on an already-installed host still points at this path,
# and those are only rewritten when the installer runs.
# =============================================================================
set -euo pipefail
STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${STACK_DIR}/scripts/start-proxy.sh" llm-a-proxy "$@"
