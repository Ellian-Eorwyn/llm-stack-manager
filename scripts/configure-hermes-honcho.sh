#!/usr/bin/env bash
# Configure vendored Hermes to use the local Honcho server.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"

SERVICE_USER="${SERVICE_USER:-$(stat -c '%U' "${STACK_DIR}")}"
SERVICE_GROUP="${SERVICE_GROUP:-$(stat -c '%G' "${STACK_DIR}")}"
USER_HOME="$(getent passwd "${SERVICE_USER}" | cut -d: -f6)"
HERMES_HOME="${HERMES_HOME:-${USER_HOME}/.hermes}"
HONCHO_CONFIG="${HERMES_HOME}/honcho.json"
HERMES_CONFIG="${HERMES_HOME}/config.yaml"

install -d -m 700 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" "${HERMES_HOME}"

sudo -u "${SERVICE_USER}" python3 - "${HONCHO_CONFIG}" "${HONCHO_URL}" "${HONCHO_WORKSPACE}" "${HONCHO_USER_PEER}" "${HONCHO_AI_PEER}" <<'PYHONCHO'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
base_url, workspace, user_peer, ai_peer = sys.argv[2:6]
if path.exists():
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
else:
    data = {}

data["baseUrl"] = base_url
data["workspace"] = workspace
data["peerName"] = user_peer
hosts = data.setdefault("hosts", {})
host = hosts.setdefault("hermes", {})
host.update(
    {
        "enabled": True,
        "workspace": workspace,
        "peerName": user_peer,
        "aiPeer": ai_peer,
        "pinPeerName": True,
        "recallMode": "hybrid",
        "writeFrequency": "async",
        "sessionStrategy": "per-repo",
        "saveMessages": True,
        "observationMode": "directional",
        "dialecticReasoningLevel": "low",
        "dialecticDepth": 1,
    }
)
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PYHONCHO
chmod 600 "${HONCHO_CONFIG}"
chown "${SERVICE_USER}:${SERVICE_GROUP}" "${HONCHO_CONFIG}"

if [[ -f "${HERMES_CONFIG}" ]]; then
    sudo -u "${SERVICE_USER}" python3 - "${HERMES_CONFIG}" <<'PYCFG'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
out = []
in_memory = False
provider_written = False
memory_seen = False

for line in lines:
    stripped = line.strip()
    top_level = bool(line) and not line.startswith((" ", "\t")) and stripped.endswith(":")
    if top_level:
        if in_memory and not provider_written:
            out.append("  provider: honcho")
            provider_written = True
        in_memory = stripped == "memory:"
        memory_seen = memory_seen or in_memory
    if in_memory and stripped.startswith("provider:"):
        out.append("  provider: honcho")
        provider_written = True
        continue
    out.append(line)

if in_memory and not provider_written:
    out.append("  provider: honcho")
elif not memory_seen:
    out.extend(["memory:", "  provider: honcho"])

path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
PYCFG
else
    sudo -u "${SERVICE_USER}" tee "${HERMES_CONFIG}" >/dev/null <<'EOF'
memory:
  provider: honcho
EOF
fi

chmod 600 "${HERMES_CONFIG}"
chown "${SERVICE_USER}:${SERVICE_GROUP}" "${HERMES_CONFIG}"

echo "Configured Hermes Honcho memory at ${HONCHO_CONFIG}"
