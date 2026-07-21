#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash $0" >&2
  exit 1
fi

mkdir -p /etc/nginx/default.d /etc/nginx/sites-available /etc/nginx/sites-enabled
cat > /etc/nginx/sites-available/llm-stack-manager <<'NGINX'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    location = / {
        default_type application/json;
        return 200 '{"service":"llm-stack-manager","status":"ok"}\n';
    }

    include /etc/nginx/default.d/*.conf;
}
NGINX

# Ubuntu's package default occupies default_server. Preserve its source file but
# disable only its generated symlink so the dedicated stack site owns port 80.
if [[ -L /etc/nginx/sites-enabled/default ]]; then
  unlink /etc/nginx/sites-enabled/default
fi
ln -sfn /etc/nginx/sites-available/llm-stack-manager /etc/nginx/sites-enabled/llm-stack-manager
nginx -t
systemctl enable --now nginx
