#!/usr/bin/env bash
# Restore systemd units and service state from backup-active-stack.sh.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/cross-platform.sh
source "${STACK_DIR}/scripts/cross-platform.sh"

if is_mac; then
    echo "rollback-to-backup.sh is Linux-only (requires systemd)." >&2
    echo "On macOS, services are managed via launchd plists in /Library/LaunchDaemons/." >&2
    exit 1
fi

# shellcheck source=scripts/stack-services.sh
source "${STACK_DIR}/scripts/stack-services.sh"

BACKUP_DIR="${1:-}"
if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo so systemd units can be restored." >&2
  exit 1
fi
if [[ -z "${BACKUP_DIR}" ]]; then
  echo "Usage: sudo bash ${STACK_DIR}/scripts/rollback-to-backup.sh /path/to/pre-cutover-backup" >&2
  exit 1
fi
if [[ ! -d "${BACKUP_DIR}" || ! -f "${BACKUP_DIR}/state/services.tsv" ]]; then
  echo "Not a valid backup directory: ${BACKUP_DIR}" >&2
  exit 1
fi

STOP_NEW_FIRST="${STOP_NEW_FIRST:-yes}"
START_RESTORED_ACTIVE="${START_RESTORED_ACTIVE:-yes}"

if [[ "${STOP_NEW_FIRST}" == "yes" ]]; then
  echo "Stopping current stack services..."
  for svc in "${STACK_SERVICES[@]}"; do
    if systemctl is-active --quiet "${svc}" 2>/dev/null; then
      echo "  stopping ${svc}"
      systemctl stop "${svc}" || true
    fi
  done
fi

echo "Restoring systemd unit files..."
for svc in "${STACK_SERVICES[@]}"; do
  src="${BACKUP_DIR}/systemd/${svc}.service"
  dst="/etc/systemd/system/${svc}.service"
  if [[ -f "${src}" ]]; then
    cp -a "${src}" "${dst}"
    chmod 644 "${dst}"
    echo "  restored ${svc}.service"
  else
    if [[ -f "${dst}" ]]; then
      rm -f "${dst}"
      echo "  removed ${svc}.service (not present in backup)"
    fi
  fi
done

systemctl daemon-reload
systemctl reset-failed "${STACK_SERVICES[@]}" 2>/dev/null || true

echo "Restoring enabled/disabled state..."
while IFS=$'\t' read -r svc enabled active loadstate fragment; do
  [[ "${svc}" == "# service enabled active loadstate fragment" ]] && continue
  [[ -z "${svc}" ]] && continue
  case "${enabled}" in
    enabled|enabled-runtime|linked|linked-runtime|alias|static|generated|indirect)
      if [[ -f "/etc/systemd/system/${svc}.service" || "$(systemctl show "${svc}" --property=LoadState --value 2>/dev/null || true)" != "not-found" ]]; then
        systemctl enable "${svc}" >/dev/null 2>&1 || true
      fi
      ;;
    disabled|masked|not-found|unknown|"")
      systemctl disable "${svc}" >/dev/null 2>&1 || true
      ;;
  esac
done < "${BACKUP_DIR}/state/services.tsv"

if [[ "${START_RESTORED_ACTIVE}" == "yes" ]]; then
  echo "Starting services that were active in the backup..."
  while IFS=$'\t' read -r svc enabled active loadstate fragment; do
    [[ "${svc}" == "# service enabled active loadstate fragment" ]] && continue
    [[ -z "${svc}" ]] && continue
    if [[ "${active}" == "active" ]]; then
      if [[ -f "/etc/systemd/system/${svc}.service" || "$(systemctl show "${svc}" --property=LoadState --value 2>/dev/null || true)" != "not-found" ]]; then
        echo "  starting ${svc}"
        systemctl start "${svc}" || true
      fi
    fi
  done < "${BACKUP_DIR}/state/services.tsv"
fi

OLD_STACK_DIR="$(cat "${BACKUP_DIR}/state/old-stack-dir.txt" 2>/dev/null || true)"
cat <<EOF
Rollback complete.
Restored backup: ${BACKUP_DIR}
Old stack path: ${OLD_STACK_DIR:-unknown}

Check status:
  systemctl status llm-manager chat-backend-dense chat-proxy embed rerank task --no-pager
EOF
