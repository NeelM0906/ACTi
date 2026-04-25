#!/bin/bash
# Step 5: copy platform code + system prompt + nginx config + status assets
# to their runtime locations. Idempotent.
set -e
echo "=== [05] install platform artifacts ==="

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

# Runtime install location
sudo mkdir -p /opt/acti/{proxy,system_prompts,ui,status,inference}
sudo cp "$REPO_ROOT/platform/proxy/sohn_proxy.py"           /opt/acti/proxy/
sudo cp "$REPO_ROOT/platform/proxy/launch_proxy.sh"         /opt/acti/proxy/
sudo cp "$REPO_ROOT/platform/system_prompts/sohn.txt"       /opt/acti/system_prompts/
sudo cp "$REPO_ROOT/platform/ui/launch_owui.sh"             /opt/acti/ui/
sudo cp "$REPO_ROOT/platform/ui/owui-branding.css"          /opt/acti/ui/
sudo cp "$REPO_ROOT/platform/inference/launch_sohn.sh"      /opt/acti/inference/
sudo cp "$REPO_ROOT/platform/status/status_collector.py"    /opt/acti/status/
sudo cp "$REPO_ROOT/platform/status/launch_status.sh"       /opt/acti/status/
sudo chmod +x /opt/acti/*/launch_*.sh

# State dirs
sudo mkdir -p /var/lib/acti /var/log/acti /var/lib/acti/openwebui /var/lib/acti/hf-cache
[ -f /var/lib/acti/api-keys.txt ] || sudo touch /var/lib/acti/api-keys.txt
sudo chmod 600 /var/lib/acti/api-keys.txt

# Status page (nginx-readable)
sudo mkdir -p /usr/share/nginx/html/acti-status
sudo cp "$REPO_ROOT/platform/status/status.html" /usr/share/nginx/html/acti-status/
[ -f /usr/share/nginx/html/acti-status/status_history.json ] || \
  echo "{\"points\":[],\"updated\":0}" | sudo tee /usr/share/nginx/html/acti-status/status_history.json >/dev/null
sudo chown -R www-data:www-data /usr/share/nginx/html/acti-status

# nginx config
sudo cp "$REPO_ROOT/platform/nginx/nginx.conf" /etc/nginx/nginx.conf
sudo nginx -t

echo "[05] artifacts installed under /opt/acti and /var/lib/acti."
