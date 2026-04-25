#!/bin/bash
# Generate a new ACTi API key and append it to the keys file.
# Usage: sudo bash scripts/generate_api_key.sh [label]
set -e
KEYFILE="${ACTI_API_KEY_FILE:-/var/lib/acti/api-keys.txt}"
LABEL="${1:-}"

KEY="sk-sohn-$(openssl rand -hex 16)"

sudo mkdir -p "$(dirname "$KEYFILE")"
[ -f "$KEYFILE" ] || sudo touch "$KEYFILE"

if [ -n "$LABEL" ]; then
  echo "# $LABEL  ($(date -Is))" | sudo tee -a "$KEYFILE" > /dev/null
fi
echo "$KEY" | sudo tee -a "$KEYFILE" > /dev/null
sudo chmod 600 "$KEYFILE"

echo "$KEY"
echo
echo "Saved to $KEYFILE."
echo "The proxy hot-reloads keys on restart: tmux kill-session -t acti-proxy && /opt/acti/proxy/launch_proxy.sh &"
