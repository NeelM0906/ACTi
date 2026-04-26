#!/bin/bash
# Step 5: copy platform code + system prompt + nginx config + status assets
# to their runtime locations. Idempotent.
set -e
echo "=== [05] install platform artifacts ==="

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

# Runtime install location
sudo mkdir -p /opt/acti/{proxy,system_prompts,ui,status,inference,skills}
sudo cp "$REPO_ROOT/platform/proxy/gateway.py"              /opt/acti/proxy/
sudo cp "$REPO_ROOT/platform/proxy/spark.py"                /opt/acti/proxy/
sudo cp "$REPO_ROOT/platform/proxy/cortex.py"               /opt/acti/proxy/
sudo cp "$REPO_ROOT/platform/proxy/cortex_prompts.py"       /opt/acti/proxy/
sudo cp "$REPO_ROOT/platform/proxy/skills.py"               /opt/acti/proxy/
sudo cp "$REPO_ROOT/platform/proxy/media.py"                /opt/acti/proxy/
sudo cp "$REPO_ROOT/platform/proxy/library.py"              /opt/acti/proxy/
sudo cp "$REPO_ROOT/platform/proxy/launch_proxy.sh"         /opt/acti/proxy/
sudo cp "$REPO_ROOT/platform/system_prompts/sohn.txt"       /opt/acti/system_prompts/
sudo cp "$REPO_ROOT/platform/ui/launch_owui.sh"             /opt/acti/ui/
sudo cp "$REPO_ROOT/platform/inference/launch_sohn.sh"      /opt/acti/inference/
sudo cp "$REPO_ROOT/platform/inference/launch_sglang.sh"    /opt/acti/inference/
sudo cp "$REPO_ROOT/platform/status/status_collector.py"    /opt/acti/status/
sudo cp "$REPO_ROOT/platform/status/launch_status.sh"       /opt/acti/status/
# Skill library — discovered by the proxy on startup. Each subdir must contain
# a SKILL.md with a `name:` and `description:` in its YAML front matter.
# _sync.py + launch_skill_sync.sh mirror the file system into the acti-ui
# `skill` table so skills authored on disk show up in Workspace -> Skills.
sudo rm -rf /opt/acti/skills
sudo cp -r "$REPO_ROOT/platform/skills"                     /opt/acti/skills

sudo chmod +x /opt/acti/*/launch_*.sh \
              /opt/acti/skills/launch_skill_sync.sh

# State dirs
sudo mkdir -p /var/lib/acti /var/log/acti /var/lib/acti/openwebui \
              /var/lib/acti/hf-cache /var/lib/acti/media /var/lib/acti/memory
[ -f /var/lib/acti/api-keys.txt ] || sudo touch /var/lib/acti/api-keys.txt
sudo chmod 600 /var/lib/acti/api-keys.txt
# Media dir must be readable by nginx (which runs as www-data on Ubuntu).
sudo chmod 755 /var/lib/acti/media

# Status page (nginx-readable)
sudo mkdir -p /usr/share/nginx/html/acti-status
sudo cp "$REPO_ROOT/platform/status/status.html" /usr/share/nginx/html/acti-status/
[ -f /usr/share/nginx/html/acti-status/status_history.json ] || \
  echo "{\"points\":[],\"updated\":0}" | sudo tee /usr/share/nginx/html/acti-status/status_history.json >/dev/null
sudo chown -R www-data:www-data /usr/share/nginx/html/acti-status

# Default-model trigger: every new chat-UI user gets Sohn pre-selected as
# their default model. We set the trigger ON the user table so it fires
# regardless of whether the user is created via the signup form or via
# the helper script we use for ops-managed accounts.
ACTI_OWUI_DB="${ACTI_OWUI_DB:-/var/lib/acti/openwebui/webui.db}"
if [ -f "$ACTI_OWUI_DB" ]; then
  sudo sqlite3 "$ACTI_OWUI_DB" <<'EOSQL'
DROP TRIGGER IF EXISTS acti_default_user_model;
CREATE TRIGGER acti_default_user_model
AFTER INSERT ON user
WHEN NEW.settings IS NULL OR json_extract(NEW.settings, '$.ui.models') IS NULL
BEGIN
  UPDATE user
  SET settings = json_set(
        COALESCE(NEW.settings, json_object()),
        '$.ui',
        json_set(
          COALESCE(json_extract(NEW.settings, '$.ui'), json_object()),
          '$.models', json_array('Sohn')
        )
      )
  WHERE id = NEW.id;
END;
EOSQL
  echo "  acti_default_user_model trigger installed in $ACTI_OWUI_DB"
fi

# nginx config
sudo cp "$REPO_ROOT/platform/nginx/nginx.conf" /etc/nginx/nginx.conf
sudo nginx -t

echo "[05] artifacts installed under /opt/acti and /var/lib/acti."
