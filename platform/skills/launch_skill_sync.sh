#!/bin/bash
# Daemon that mirrors platform/skills/<name>/SKILL.md into the acti-ui
# `skill` table on a polling loop. File system is the source of truth.
# Runs under tmux as the `acti-skill-sync` service.
set -e
set -o pipefail

ACTI_SKILLS_DIR="${ACTI_SKILLS_DIR:-/opt/acti/skills}"
ACTI_LOG_DIR="${ACTI_LOG_DIR:-/var/log/acti}"
ACTI_SKILL_SYNC_INTERVAL="${ACTI_SKILL_SYNC_INTERVAL:-5}"
# Default DB path matches the OWUI launcher's DATA_DIR
# (platform/ui/launch_owui.sh sets DATA_DIR=/var/lib/acti/openwebui).
OWUI_DB="${OWUI_DB:-/var/lib/acti/openwebui/webui.db}"

mkdir -p "$ACTI_LOG_DIR"
LOG="$ACTI_LOG_DIR/skill-sync.log"

export ACTI_SKILLS_DIR ACTI_SKILL_SYNC_INTERVAL OWUI_DB

echo "[$(date -Is)] launch_skill_sync.sh starting (every ${ACTI_SKILL_SYNC_INTERVAL}s, dir=$ACTI_SKILLS_DIR, db=$OWUI_DB)" \
  | tee -a "$LOG"

# stdlib only — no conda env needed
exec python3 -u /opt/acti/skills/_sync.py 2>&1 | tee -a "$LOG"
