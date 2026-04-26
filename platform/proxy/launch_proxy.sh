#!/bin/bash
# ACTi Sohn API proxy launcher.
set -e
set -o pipefail

ACTI_PYTHON_ENV="${ACTI_PYTHON_ENV:-acti-inference}"
ACTI_PROXY_PORT="${SOHN_PROXY_PORT:-8080}"
ACTI_LOG_DIR="${ACTI_LOG_DIR:-/var/log/acti}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$ACTI_PYTHON_ENV"

# Resolve script directory for proxy module path
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mkdir -p "$ACTI_LOG_DIR"
LOG="$ACTI_LOG_DIR/proxy.log"
echo "[$(date -Is)] proxy launching on :$ACTI_PROXY_PORT" | tee -a "$LOG"

export SOHN_PROXY_PORT="$ACTI_PROXY_PORT"

# Optional: load tool credentials from non-tracked env files.
# Keep these OUTSIDE the repo. Format: shell-style `KEY=VALUE` lines.
[ -f /etc/acti/media.env ]   && set -a && . /etc/acti/media.env   && set +a
[ -f /etc/acti/library.env ] && set -a && . /etc/acti/library.env && set +a

exec python "$SCRIPT_DIR/gateway.py" 2>&1 | tee -a "$LOG"
