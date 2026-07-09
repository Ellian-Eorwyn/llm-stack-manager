#!/usr/bin/env bash
# Stop the Playwright WS server
set -euo pipefail
PIDFILE="$(dirname "$0")/logs/server.pid"
if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping playwright server (PID $PID)..."
    kill "$PID"
    rm -f "$PIDFILE"
  else
    echo "Process $PID not running. Cleaning up."
    rm -f "$PIDFILE"
  fi
else
  echo "No PID file found. Trying to kill by port..."
  lsof -ti:3001 | xargs -r kill
fi
