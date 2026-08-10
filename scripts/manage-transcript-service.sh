#!/usr/bin/env bash
# =============================================================================
# manage-transcript-service.sh
# Start/stop the transcription sidecar without systemd.
#
# `app.should_use_local_transcript_manager` picks this path when no
# transcript-backend unit is installed, so the sidecar can be driven from the
# manager UI on a host where the stack was never installed as root — and so it
# can be exercised at all without sudo.
# =============================================================================
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${STACK_DIR}/logs/transcript"
LOG_FILE="${LOG_DIR}/transcript-backend.log"
PID_FILE="${LOG_DIR}/transcript-backend.pid"

mkdir -p "${LOG_DIR}"

running_pid() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    [[ -n "${pid}" ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    printf '%s' "${pid}"
}

start_service() {
    if pid="$(running_pid)"; then
        echo "[transcribe] Already running (pid ${pid})"
        return 0
    fi
    nohup bash "${STACK_DIR}/scripts/start-transcribe.sh" >>"${LOG_FILE}" 2>&1 &
    local pid=$!
    echo "${pid}" > "${PID_FILE}"
    sleep 1
    if ! kill -0 "${pid}" 2>/dev/null; then
        rm -f "${PID_FILE}"
        echo "[transcribe] Failed to start; see ${LOG_FILE}" >&2
        tail -n 20 "${LOG_FILE}" >&2 || true
        return 1
    fi
    echo "[transcribe] Started (pid ${pid}), logging to ${LOG_FILE}"
}

stop_service() {
    if ! pid="$(running_pid)"; then
        rm -f "${PID_FILE}"
        echo "[transcribe] Not running"
        return 0
    fi
    # The launcher execs the python process, so the recorded pid is the server.
    kill "${pid}" 2>/dev/null || true
    for _ in $(seq 1 40); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.25
    done
    kill -9 "${pid}" 2>/dev/null || true
    rm -f "${PID_FILE}"
    echo "[transcribe] Stopped (pid ${pid})"
}

status_service() {
    if pid="$(running_pid)"; then
        echo "[transcribe] running (pid ${pid})"
        return 0
    fi
    echo "[transcribe] stopped"
    return 3
}

case "${1:-status}" in
    start)   start_service ;;
    stop)    stop_service ;;
    restart) stop_service; start_service ;;
    status)  status_service ;;
    *) echo "Usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
