#!/bin/bash
# Daemon that mirrors Cortex-managed memory files (/var/lib/acti/memory/*.md)
# into the acti-ui `memory` table on a polling loop. File system is the source
# of truth. Runs under tmux as the `acti-memory-sync` service.
set -e
set -o pipefail

ACTI_MEMORY_DIR="${ACTI_MEMORY_DIR:-/var/lib/acti/memory}"
ACTI_LOG_DIR="${ACTI_LOG_DIR:-/var/log/acti}"
ACTI_MEMORY_SYNC_INTERVAL="${ACTI_MEMORY_SYNC_INTERVAL:-10}"
# Default DB path matches the acti-ui launcher's DATA_DIR.
OWUI_DB="${OWUI_DB:-/var/lib/acti/openwebui/webui.db}"

mkdir -p "$ACTI_LOG_DIR"
LOG="$ACTI_LOG_DIR/memory-sync.log"

export ACTI_MEMORY_DIR ACTI_MEMORY_SYNC_INTERVAL OWUI_DB

echo "[$(date -Is)] launch_memory_sync.sh starting (every ${ACTI_MEMORY_SYNC_INTERVAL}s, dir=$ACTI_MEMORY_DIR, db=$OWUI_DB)" \
  | tee -a "$LOG"

# stdlib only — no conda env needed
exec python3 -u /opt/acti/memories/_sync.py 2>&1 | tee -a "$LOG"
