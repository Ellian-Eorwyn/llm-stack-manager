#!/usr/bin/env bash
# Expose chat-proxy2's aggregate think/chat/code endpoint through nginx.
set -euo pipefail

export AGGREGATE_PUBLIC_PORT="${AGGREGATE_PUBLIC_PORT:-8020}"
export AGGREGATE_UPSTREAM_PORT="${AGGREGATE_UPSTREAM_PORT:-8112}"
export AGGREGATE_CONF_NAME="${AGGREGATE_CONF_NAME:-llm-aggregate2-${AGGREGATE_PUBLIC_PORT}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/install-aggregate-nginx.sh"
