#!/bin/bash
# Step 4: in-place patch to OpenWebUI to remove the hardcoded "(Open WebUI)"
# suffix from app name strings.
#
# Without this patch, the onboarding screen, login screen, and tab title all
# end with "(Open WebUI)" regardless of WEBUI_NAME. Reverting just requires
# reinstalling open-webui from pip.
set -e
ENV_NAME="${ACTI_PYTHON_ENV:-acti-inference}"
PKG="/opt/conda/envs/$ENV_NAME/lib/python3.12/site-packages/open_webui"

[ -d "$PKG" ] || { echo "  open_webui not installed in env $ENV_NAME — skipping"; exit 0; }

# Backup once
[ -f "$PKG/.acti_backup_done" ] || {
  cp "$PKG/env.py" "$PKG/env.py.bak"
  touch "$PKG/.acti_backup_done"
  echo "  backed up env.py"
}

# Comment out the suffix-append branch
sed -i "s|if WEBUI_NAME != 'Open WebUI':|if False:  # ACTi: suppress \" (Open WebUI)\" suffix|" "$PKG/env.py"

# Drop pycache so the patched module is re-imported
find "$PKG" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "[04] OpenWebUI patched."
