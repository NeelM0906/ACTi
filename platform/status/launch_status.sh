#!/bin/bash
# ACTi status collector launcher.
set -e
ACTI_PYTHON_ENV="${ACTI_PYTHON_ENV:-acti-inference}"
ACTI_LOG_DIR="${ACTI_LOG_DIR:-/var/log/acti}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$ACTI_PYTHON_ENV"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
mkdir -p "$ACTI_LOG_DIR"

exec python "$SCRIPT_DIR/status_collector.py" 2>&1 | tee -a "$ACTI_LOG_DIR/status.log"
