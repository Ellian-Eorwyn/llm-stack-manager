#!/usr/bin/env bash
# Snapshot current LLM stack systemd/config state before installing this repo.
# This does not stop services and does not modify the old stack.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/stack-services.sh
source "${STACK_DIR}/scripts/stack-services.sh"

OLD_STACK_DIR="${1:-/mnt/LLMs/llamacpp/llm-stack}"
BACKUP_ROOT="${LLM_STACK_BACKUP_ROOT:-${STACK_DIR}/backups}"
TS="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT}/pre-cutover-${TS}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo so systemd unit files and service state can be captured." >&2
  exit 1
fi

mkdir -p \
  "${BACKUP_DIR}/systemd" \
  "${BACKUP_DIR}/config" \
  "${BACKUP_DIR}/logs" \
  "${BACKUP_DIR}/state"

readlink -f "${OLD_STACK_DIR}" > "${BACKUP_DIR}/state/old-stack-dir.txt" 2>/dev/null || printf '%s\n' "${OLD_STACK_DIR}" > "${BACKUP_DIR}/state/old-stack-dir.txt"
printf '%s\n' "${STACK_DIR}" > "${BACKUP_DIR}/state/new-stack-dir.txt"
date --iso-8601=seconds > "${BACKUP_DIR}/state/created-at.txt"

{
  echo "# service enabled active loadstate fragment"
  for svc in "${STACK_SERVICES[@]}"; do
    enabled="$(systemctl is-enabled "${svc}" 2>/dev/null || true)"
    active="$(systemctl is-active "${svc}" 2>/dev/null || true)"
    loadstate="$(systemctl show "${svc}" --property=LoadState --value 2>/dev/null || true)"
    fragment="$(systemctl show "${svc}" --property=FragmentPath --value 2>/dev/null || true)"
    printf '%s\t%s\t%s\t%s\t%s\n' "${svc}" "${enabled:-unknown}" "${active:-unknown}" "${loadstate:-unknown}" "${fragment:-}"

    if [[ -n "${fragment}" && -f "${fragment}" ]]; then
      cp -a "${fragment}" "${BACKUP_DIR}/systemd/${svc}.service"
    fi
  done
} > "${BACKUP_DIR}/state/services.tsv"

systemctl list-unit-files 'llm-manager.service' 'chat-*.service' 'embed.service' 'rerank.service' 'task.service' 'think.service' 'nothink.service' 'qwen-*.service' 'honcho-*.service' 'graphiti.service' 'ocr.service' 'glmocr-sdk.service' 'transcript-backend.service' 'tts-*.service' > "${BACKUP_DIR}/state/unit-files.txt" 2>&1 || true
systemctl list-units --all 'llm-manager.service' 'chat-*.service' 'embed.service' 'rerank.service' 'task.service' 'think.service' 'nothink.service' 'qwen-*.service' 'honcho-*.service' 'graphiti.service' 'ocr.service' 'glmocr-sdk.service' 'transcript-backend.service' 'tts-*.service' > "${BACKUP_DIR}/state/units.txt" 2>&1 || true

for svc in "${STACK_SERVICES[@]}"; do
  systemctl cat "${svc}" > "${BACKUP_DIR}/systemd/${svc}.cat.txt" 2>&1 || true
  systemctl status --no-pager --lines=20 "${svc}" > "${BACKUP_DIR}/logs/${svc}.status.txt" 2>&1 || true
done

if [[ -d "${OLD_STACK_DIR}/config" ]]; then
  cp -a "${OLD_STACK_DIR}/config/." "${BACKUP_DIR}/config/"
fi

if [[ -f "${OLD_STACK_DIR}/install.sh" ]]; then
  cp -a "${OLD_STACK_DIR}/install.sh" "${BACKUP_DIR}/install.sh"
fi
if [[ -d "${OLD_STACK_DIR}/scripts" ]]; then
  find "${OLD_STACK_DIR}/scripts" -maxdepth 1 -type f -printf '%f\n' | sort > "${BACKUP_DIR}/state/old-scripts.txt"
fi

cat > "${BACKUP_DIR}/README.txt" <<EOF
LLM stack pre-cutover backup
Created: $(cat "${BACKUP_DIR}/state/created-at.txt")
Old stack: $(cat "${BACKUP_DIR}/state/old-stack-dir.txt")
New stack: ${STACK_DIR}

Rollback command:
  sudo bash ${STACK_DIR}/scripts/rollback-to-backup.sh ${BACKUP_DIR}

This backup restores systemd unit files and service enabled/active state. It does
not delete the new stack directory and does not rewrite the old stack directory.
EOF

chmod -R go-rwx "${BACKUP_DIR}"
printf '%s\n' "${BACKUP_DIR}"
echo "Backup complete: ${BACKUP_DIR}"
echo "Rollback command: sudo bash ${STACK_DIR}/scripts/rollback-to-backup.sh ${BACKUP_DIR}"
