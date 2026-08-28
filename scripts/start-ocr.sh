#!/usr/bin/env bash
# =============================================================================
# start-ocr.sh
# Shim. This slot is served by scripts/start-backend.sh, which builds the
# command from web/backends/slots.py.
#
# Kept as a name rather than deleted because a generated systemd unit or launchd
# plist on an already-installed host still points at this path, and those are
# only rewritten when the installer runs. Removing it would break a running
# stack on the next restart, before anyone re-ran install.sh. It goes once the
# installed units name start-backend.sh.
# =============================================================================
set -euo pipefail
STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${STACK_DIR}/scripts/start-backend.sh" ocr "$@"
