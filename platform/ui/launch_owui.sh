#!/bin/bash
# acti-ui — the ACTi chat UI. Wired to the Sohn proxy via the OpenAI
# compatibility surface. Branding lives natively in the fork at
# vendor/acti-ui.
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

# --- Branding ---
# Default app name in the fork is already "ACTi"; override here only if you
# want a deployment-specific suffix (e.g. "ACTi (staging)").
# export WEBUI_NAME="ACTi"

# --- Auth ---
export WEBUI_AUTH=true
export DEFAULT_MODELS="Sohn"
# Forward signed-in user identity to the upstream (Sohn proxy) as request
# headers — the proxy uses X-OpenWebUI-User-Id to partition cortex memory
# per user (preventing cross-account identity leaks) and X-OpenWebUI-User-Name
# to seed the user's signup-collected name so the model can address them by
# name from their first turn. MUST stay on for memory isolation to work.
export ENABLE_FORWARD_USER_INFO_HEADERS=true

# Override OWUI's default chat-title prompt. The default ships with
# "Generate a concise, 3-5 word title with an emoji ..." — Sohn's branding
# is no-emoji, so we strip that clause. The proxy adds a defense-in-depth
# style guard for ALL auxiliary calls, but overriding here at the source
# keeps OWUI's instruction internally consistent.
export TITLE_GENERATION_PROMPT_TEMPLATE='### Task:
Generate a concise 3-5 word title summarizing the chat history.

### Guidelines:
- The title should clearly represent the main theme or subject of the conversation.
- Use plain text only — NO emojis, NO quotation marks, NO special formatting.
- Write the title in the chat'"'"'s primary language; default to English if multilingual.
- Length must be between 3 and 5 words.

### Output:
JSON format: { "title": "your concise title here" }

### Chat History:
<chat_history>
{{MESSAGES:END:6}}
</chat_history>'
# DEFAULT_USER_ROLE intentionally NOT set so the first signup becomes admin.
# Future signups default to "pending" via the admin panel after first admin exists.
# Admins see ALL resources regardless of access grants — without this, OWUI 0.9.2
# requires an explicit access_grant row per resource per user, which our
# platform-managed skill / tool sync daemons don't create. For our 4-10 person
# team where the admin manages the catalog, bypass is the right default.
export BYPASS_ADMIN_ACCESS_CONTROL=true
# Same idea, but for the model access filter — Sohn is a backend-fetched
# model with no preset row, so without bypass the per-user filter at
# routers/openai.get_filtered_models hides it from non-admin users
# entirely (only admins fall through the elif user.role == 'admin'
# branch). For our team model this is the right default.
export BYPASS_MODEL_ACCESS_CONTROL=true

# Static-asset dir override — the chat UI's branding files (favicon,
# user.png, splash images) are produced by the SvelteKit build into
# `frontend/static/` and we want the FastAPI /static mount to serve from
# there so the sidebar + avatars don't render as broken images.
export STATIC_DIR=/opt/conda/envs/vllm-rocm/lib/python3.12/site-packages/open_webui/frontend

# --- Data ---
export DATA_DIR="${ACTI_OWUI_DATA_DIR:-/var/lib/acti/openwebui}"
mkdir -p "$DATA_DIR"

# --- Image generation (toggle in the chat + button) ---
# Routes through the gateway's /v1/images/generations, which calls Lumen
# server-side. ACTi-managed; no DALL-E or external service involved.
export ENABLE_IMAGE_GENERATION=true
export IMAGE_GENERATION_ENGINE=openai
export IMAGES_OPENAI_API_BASE_URL="http://127.0.0.1:8080/v1"
export IMAGES_OPENAI_API_KEY="$OPENAI_API_KEY"
# Default size and model — Lumen accepts size as WxH and ignores model name.
export IMAGE_SIZE="1024x1024"
export IMAGE_GENERATION_MODEL="lumen"

# --- Video generation (toggle in the chat + button) ---
# ACTi-only: no native OWUI feature. Custom handler in the fork's
# middleware calls our gateway's /v1/videos/generations endpoint.
export ENABLE_VIDEO_GENERATION=true
export VIDEO_GENERATION_API_BASE_URL="http://127.0.0.1:8080/v1"
export VIDEO_GENERATION_API_KEY="$OPENAI_API_KEY"

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
