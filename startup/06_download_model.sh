#!/bin/bash
# Step 6: download base model weights into HF cache.
# Reads ACTI_MODEL_ID and HF_TOKEN from .env (or environment).
set -e
echo "=== [06] download model weights ==="

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
[ -f "$REPO_ROOT/startup/.env" ] && set -a && . "$REPO_ROOT/startup/.env" && set +a

: "${ACTI_MODEL_ID:?set ACTI_MODEL_ID in startup/.env}"
: "${HF_TOKEN:?set HF_TOKEN in startup/.env}"
ENV_NAME="${ACTI_PYTHON_ENV:-acti-inference}"
ACTI_HF_HOME="${ACTI_HF_HOME:-/var/lib/acti/hf-cache}"

export HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="$ACTI_HF_HOME"
mkdir -p "$ACTI_HF_HOME"

echo "  fetching weights for: <hidden> (logged at /var/log/acti/download.log)"
mkdir -p /var/log/acti
/opt/conda/envs/$ENV_NAME/bin/hf download "$ACTI_MODEL_ID" \
  --cache-dir "$ACTI_HF_HOME" \
  --exclude "*.bin" \
  --exclude "*.pth" \
  --exclude "*.gguf" \
  --exclude "original/*" \
  --exclude "consolidated*" \
  > /var/log/acti/download.log 2>&1
echo "[06] weights downloaded."
