#!/usr/bin/env bash
# Install or update the local SearXNG instance used by LLM Stack Manager.
set -euo pipefail

STACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${STACK_DIR}/config/llm-stack.env"

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run with sudo: sudo bash ${STACK_DIR}/scripts/install-searxng.sh" >&2
    exit 1
fi

if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
fi

SEARXNG_HOME="${SEARXNG_HOME:-/usr/local/searxng}"
SEARXNG_SRC="${SEARXNG_HOME}/searxng-src"
SEARXNG_VENV="${SEARXNG_HOME}/searx-pyenv"
SEARXNG_RUN="${SEARXNG_HOME}/run"
SEARXNG_SETTINGS_PATH="${SEARXNG_SETTINGS_PATH:-/etc/searxng/settings.yml}"
SEARXNG_UWSGI_INI="${SEARXNG_UWSGI_INI:-/etc/uwsgi/apps-available/searxng.ini}"
SEARXNG_UWSGI_SOCKET="${SEARXNG_UWSGI_SOCKET:-${SEARXNG_RUN}/socket}"
SEARXNG_NGINX_CONF="${SEARXNG_NGINX_CONF:-/etc/nginx/default.apps-available/searxng.conf}"
SEARXNG_URL_PATH="${SEARXNG_URL_PATH:-/searxng}"
SEARXNG_REPO="${SEARXNG_REPO:-https://github.com/searxng/searxng}"
SEARXNG_REF="${SEARXNG_REF:-master}"

install_packages() {
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y \
            python3-dev python3-babel python3-venv python-is-python3 \
            uwsgi uwsgi-plugin-python3 nginx git build-essential \
            libxslt-dev zlib1g-dev libffi-dev libssl-dev
    else
        echo "No supported package manager found. Install SearXNG dependencies manually." >&2
        exit 1
    fi
}

ensure_user_and_dirs() {
    if ! id searxng >/dev/null 2>&1; then
        useradd --shell /bin/bash --system \
            --home-dir "${SEARXNG_HOME}" \
            --comment "Privacy-respecting metasearch engine" \
            searxng
    fi
    mkdir -p "${SEARXNG_HOME}" "${SEARXNG_RUN}" "$(dirname "${SEARXNG_SETTINGS_PATH}")" \
        /etc/uwsgi/apps-available /etc/uwsgi/apps-enabled \
        /etc/nginx/default.apps-available /etc/nginx/default.d
    chown -R searxng:searxng "${SEARXNG_HOME}"
}

install_source() {
    if [[ -d "${SEARXNG_SRC}/.git" ]]; then
        sudo -H -u searxng git -C "${SEARXNG_SRC}" fetch --tags --prune origin
        sudo -H -u searxng git -C "${SEARXNG_SRC}" checkout "${SEARXNG_REF}"
        sudo -H -u searxng git -C "${SEARXNG_SRC}" pull --ff-only origin "${SEARXNG_REF}" || true
    else
        rm -rf "${SEARXNG_SRC}"
        sudo -H -u searxng git clone "${SEARXNG_REPO}" "${SEARXNG_SRC}"
        sudo -H -u searxng git -C "${SEARXNG_SRC}" checkout "${SEARXNG_REF}"
    fi

    if [[ ! -d "${SEARXNG_VENV}" ]]; then
        sudo -H -u searxng python3 -m venv "${SEARXNG_VENV}"
    fi
    sudo -H -u searxng "${SEARXNG_VENV}/bin/python" -m pip install -U pip setuptools wheel
    sudo -H -u searxng "${SEARXNG_VENV}/bin/python" -m pip install -U pyyaml msgspec typing-extensions pybind11
    sudo -H -u searxng bash -lc "cd '${SEARXNG_SRC}' && '${SEARXNG_VENV}/bin/python' -m pip install --use-pep517 --no-build-isolation -e ."
}

write_settings() {
    local secret="${SEARXNG_SECRET:-}"
    if [[ -z "${secret}" && -f "${SEARXNG_SETTINGS_PATH}" ]]; then
        secret="$(python3 - "${SEARXNG_SETTINGS_PATH}" <<'PYPREVIOUSSECRET'
import re
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(r'^\s*secret_key:\s*["\x27]?([^"\x27\s]+)', text, re.MULTILINE)
print(match.group(1) if match else "")
PYPREVIOUSSECRET
)"
    fi
    if [[ -z "${secret}" ]]; then
        secret="$(python3 - <<'PYSECRET'
import secrets
print(secrets.token_hex(32))
PYSECRET
)"
    fi
    local formats="${SEARXNG_FORMATS:-html,json}"
    python3 - "${SEARXNG_SETTINGS_PATH}" "${secret}" "${SEARXNG_INSTANCE_NAME:-SearXNG}" \
        "${SEARXNG_SAFE_SEARCH:-2}" "${SEARXNG_AUTOCOMPLETE:-duckduckgo}" \
        "${formats}" "${SEARXNG_LIMITER:-false}" "${SEARXNG_IMAGE_PROXY:-true}" \
        "${SEARXNG_VALKEY_URL:-valkey://localhost:6379/0}" <<'PYSETTINGS'
import sys
from pathlib import Path

path, secret, instance, safe, autocomplete, formats, limiter, image_proxy, valkey = sys.argv[1:]
format_lines = "\n".join(f"    - {item.strip()}" for item in formats.split(",") if item.strip())
content = f"""# SearXNG settings

use_default_settings: true

general:
  debug: false
  instance_name: "{instance}"

search:
  safe_search: {safe}
  autocomplete: '{autocomplete}'
  formats:
{format_lines}

server:
  secret_key: "{secret}"
  limiter: {limiter}
  image_proxy: {image_proxy}

valkey:
  url: {valkey}
"""
Path(path).write_text(content, encoding="utf-8")
PYSETTINGS
    chmod 640 "${SEARXNG_SETTINGS_PATH}"
    chown root:searxng "${SEARXNG_SETTINGS_PATH}"
}

write_uwsgi() {
    cat > "${SEARXNG_UWSGI_INI}" <<UWSGI
# -*- mode: conf; coding: utf-8  -*-
[uwsgi]
uid = searxng
gid = searxng
env = LANG=C.UTF-8
env = LANGUAGE=C.UTF-8
env = LC_ALL=C.UTF-8
chdir = ${SEARXNG_SRC}/searx
env = SEARXNG_SETTINGS_PATH=${SEARXNG_SETTINGS_PATH}
disable-logging = true
chmod-socket = 666
single-interpreter = true
master = true
lazy-apps = true
plugin = python3,http
enable-threads = true
workers = %k
threads = 4
module = searx.webapp
virtualenv = ${SEARXNG_VENV}
pythonpath = ${SEARXNG_SRC}
socket = ${SEARXNG_UWSGI_SOCKET}
buffer-size = 8192
offload-threads = %k
UWSGI
    chmod 644 "${SEARXNG_UWSGI_INI}"
    ln -sfn "${SEARXNG_UWSGI_INI}" /etc/uwsgi/apps-enabled/searxng.ini
}

write_nginx() {
    cat > "${SEARXNG_NGINX_CONF}" <<NGINX
location ${SEARXNG_URL_PATH} {
    uwsgi_pass unix://${SEARXNG_UWSGI_SOCKET};
    include uwsgi_params;
    uwsgi_param HTTP_HOST \$host;
    uwsgi_param HTTP_CONNECTION \$http_connection;
    uwsgi_param HTTP_X_SCHEME \$scheme;
    uwsgi_param HTTP_X_FORWARDED_PROTO \$scheme;
    uwsgi_param HTTP_X_SCRIPT_NAME ${SEARXNG_URL_PATH};
    uwsgi_param HTTP_X_REAL_IP \$remote_addr;
    uwsgi_param HTTP_X_FORWARDED_FOR \$proxy_add_x_forwarded_for;
}
NGINX
    chmod 644 "${SEARXNG_NGINX_CONF}"
    ln -sfn "${SEARXNG_NGINX_CONF}" /etc/nginx/default.d/searxng.conf
}

restart_services() {
    systemctl daemon-reload
    systemctl enable uwsgi nginx >/dev/null 2>&1 || true
    systemctl restart uwsgi
    nginx -t
    systemctl reload nginx
}

if [[ "${SEARXNG_ENABLED:-on}" != "on" ]]; then
    echo "SearXNG is disabled in ${CONFIG_FILE}; skipping install."
    exit 0
fi

install_packages
ensure_user_and_dirs
install_source
write_settings
write_uwsgi
write_nginx
restart_services

echo "SearXNG installed/updated at ${SEARXNG_PUBLIC_URL:-${SEARXNG_BASE_URL:-http://127.0.0.1${SEARXNG_URL_PATH}/}}"
