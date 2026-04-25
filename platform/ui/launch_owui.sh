#!/bin/bash
# OpenWebUI wired to the Sohn proxy, branded as ACTi AI.
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate vllm-rocm

# --- Backend: point at Sohn proxy ---
export OPENAI_API_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="$(head -1 ${ACTI_API_KEY_FILE:-/var/lib/acti/api-keys.txt})"
export OPENAI_API_BASE_URLS="http://127.0.0.1:8080/v1"
export OPENAI_API_KEYS="$OPENAI_API_KEY"
export ENABLE_OPENAI_API=true
export ENABLE_OLLAMA_API=false

# --- Branding: ACTi AI ---
export WEBUI_NAME="ACTi AI"
export WEBUI_BANNERS='[{"id":"acti-welcome","type":"info","title":"Welcome to ACTi","content":"Powered by Sohn · v0.0.1","dismissible":true,"timestamp":1777000000}]'
# Inject ACTi CSS overrides on every page
export CUSTOM_CSS="$(cat ${ACTI_OWUI_BRANDING_CSS:-/opt/acti/ui/owui-branding.css} 2>/dev/null || echo '')"

# --- Auth ---
export WEBUI_AUTH=true
export DEFAULT_MODELS="Sohn"
# DEFAULT_USER_ROLE intentionally NOT set so the first signup becomes admin.
# Future signups default to "pending" via the admin panel after first admin exists.

# --- Data ---
export DATA_DIR="${ACTI_OWUI_DATA_DIR:-/var/lib/acti/openwebui}"
mkdir -p "$DATA_DIR"

# --- Web search (DuckDuckGo — no API key needed) ---
export ENABLE_RAG_WEB_SEARCH=true
export RAG_WEB_SEARCH_ENGINE="duckduckgo"
export WEB_SEARCH_ENGINE="duckduckgo"
export ENABLE_WEB_SEARCH=true
export RAG_WEB_SEARCH_RESULT_COUNT=5
export RAG_WEB_SEARCH_CONCURRENT_REQUESTS=10
export ENABLE_SEARCH_QUERY_GENERATION=true

# --- Disable other noise ---
export RAG_EMBEDDING_ENGINE=""
export AUDIO_STT_ENGINE=""
export AUDIO_TTS_ENGINE=""
export ENABLE_EVALUATION_ARENA_MODELS=false
export ENABLE_COMMUNITY_SHARING=false
export ENABLE_VERSION_UPDATE_CHECK=false

exec open-webui serve --host 127.0.0.1 --port 3000
