#!/usr/bin/env bash
# Expose the aggregate think/chat/code proxy through nginx.
#
# The chat proxy listens on AGGREGATE_UPSTREAM_PORT, while nginx listens on the
# Tailscale/public AGGREGATE_PUBLIC_PORT. Binding nginx to the Tailscale address
# avoids conflicts with raw llama.cpp backends on 127.0.0.1:8010/8020.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run as root: sudo bash scripts/install-aggregate-nginx.sh" >&2
    exit 1
fi

AGGREGATE_PUBLIC_PORT="${AGGREGATE_PUBLIC_PORT:-8010}"
AGGREGATE_UPSTREAM_PORT="${AGGREGATE_UPSTREAM_PORT:-8012}"
AGGREGATE_SERVER_NAME="${AGGREGATE_SERVER_NAME:-llms llms.tailfad058.ts.net _}"
AGGREGATE_LISTEN_ADDR="${AGGREGATE_LISTEN_ADDR:-auto}"
AGGREGATE_CONF_NAME="${AGGREGATE_CONF_NAME:-llm-aggregate-${AGGREGATE_PUBLIC_PORT}}"
CONF_PATH="${CONF_PATH:-/etc/nginx/sites-available/${AGGREGATE_CONF_NAME}}"
LINK_PATH="${LINK_PATH:-/etc/nginx/sites-enabled/${AGGREGATE_CONF_NAME}}"

listen_lines() {
    local addr="${AGGREGATE_LISTEN_ADDR}"
    if [[ "${addr}" == "auto" ]]; then
        addr="$(tailscale ip -4 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -n "${addr}" ]]; then
        printf '    listen %s:%s;\n' "${addr}" "${AGGREGATE_PUBLIC_PORT}"
        return
    fi
    printf '    listen %s;\n' "${AGGREGATE_PUBLIC_PORT}"
    printf '    listen [::]:%s;\n' "${AGGREGATE_PUBLIC_PORT}"
}

cat > "${CONF_PATH}" <<NGINX
server {
$(listen_lines)
    server_name ${AGGREGATE_SERVER_NAME};

    location / {
        proxy_pass http://127.0.0.1:${AGGREGATE_UPSTREAM_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
NGINX

ln -sfn "${CONF_PATH}" "${LINK_PATH}"
nginx -t
systemctl reload nginx

echo "Installed nginx aggregate proxy: ${AGGREGATE_LISTEN_ADDR}:${AGGREGATE_PUBLIC_PORT} -> 127.0.0.1:${AGGREGATE_UPSTREAM_PORT}"
