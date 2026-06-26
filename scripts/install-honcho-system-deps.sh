#!/usr/bin/env bash
# Install and initialize local PostgreSQL/pgvector and Redis for Honcho.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${STACK_DIR}/config/llm-stack.env"

HONCHO_ENV_FILE="${HONCHO_ENV_FILE:-${STACK_DIR}/config/honcho.env}"
if [[ ! -f "${HONCHO_ENV_FILE}" ]]; then
    echo "Missing ${HONCHO_ENV_FILE}; run install.sh so it can be generated first." >&2
    exit 1
fi

set -a
# shellcheck source=/dev/null
source "${HONCHO_ENV_FILE}"
set +a

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run with sudo: sudo bash ${STACK_DIR}/scripts/install-honcho-system-deps.sh" >&2
    exit 1
fi

SERVICE_USER="${SERVICE_USER:-$(stat -c '%U' "${STACK_DIR}")}"

install_packages() {
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y postgresql postgresql-server-dev-all redis-server build-essential git make python3-venv python3-pip
    else
        echo "No supported package manager found. Install PostgreSQL, pgvector, and Redis manually." >&2
        exit 1
    fi
}

ensure_pgvector() {
    if sudo -u postgres psql -tAc "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'" | grep -qx "1"; then
        return
    fi

    local pgvector_dir="${STACK_DIR}/deps/pgvector"
    if [[ ! -d "${pgvector_dir}/.git" ]]; then
        sudo -u "${SERVICE_USER}" git clone https://github.com/pgvector/pgvector.git "${pgvector_dir}"
    else
        sudo -u "${SERVICE_USER}" git -C "${pgvector_dir}" pull --ff-only
    fi
    make -C "${pgvector_dir}"
    make -C "${pgvector_dir}" install
    chown -R "${SERVICE_USER}:$(stat -c '%G' "${STACK_DIR}")" "${pgvector_dir}"
}

db_password_from_uri() {
    python3 - "${DB_CONNECTION_URI}" <<'PYDB'
from urllib.parse import urlparse, unquote
import sys
parsed = urlparse(sys.argv[1])
print(unquote(parsed.password or ""))
PYDB
}

sql_literal() {
    python3 - "$1" <<'PYSQL'
import sys
print(sys.argv[1].replace("'", "''"))
PYSQL
}

install_packages
systemctl enable --now postgresql
systemctl enable --now redis-server
ensure_pgvector

DB_PASSWORD="$(db_password_from_uri)"
if [[ -z "${DB_PASSWORD}" ]]; then
    echo "DB_CONNECTION_URI in ${HONCHO_ENV_FILE} must include a password." >&2
    exit 1
fi
DB_PASSWORD_SQL="$(sql_literal "${DB_PASSWORD}")"

sudo -u postgres psql <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'honcho') THEN
    CREATE ROLE honcho LOGIN PASSWORD '${DB_PASSWORD_SQL}';
  ELSE
    ALTER ROLE honcho WITH LOGIN PASSWORD '${DB_PASSWORD_SQL}';
  END IF;
END
\$\$;
SELECT 'CREATE DATABASE honcho OWNER honcho'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'honcho')\gexec
SQL

sudo -u postgres psql -d honcho <<SQL
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL PRIVILEGES ON DATABASE honcho TO honcho;
SQL

echo "Honcho local PostgreSQL/pgvector and Redis are ready."
