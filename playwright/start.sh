#!/usr/bin/env bash
# Start the Playwright WS server
# Usage: ./start.sh [--port 3001]
set -euo pipefail
cd "$(dirname "$0")"
exec node server.js "$@"
