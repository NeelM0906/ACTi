# ACTi Platform — Architecture

The platform is a layered stack. Each layer can be operated, restarted, and debugged independently.

```
┌─────────────────────────────────────────────────────────────┐
│                       PUBLIC SURFACE                        │
│                                                             │
│   nginx :8888  ──►  /              → ACTi chat UI           │
│                ──►  /v1/*          → Sohn API (with persona)│
│                ──►  /raw/v1/*      → Raw LLM passthrough    │
│                ──►  /raw/*         → Same; alias for clients│
│                                      that strip /v1         │
│                ──►  /status        → Status page            │
│                ──►  /sohn-health   → Liveness probe         │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
   ┌───────▼─────────┐          ┌─────────▼──────────┐
   │  ACTi Chat UI   │          │  ACTi Gateway      │
   │  (acti-ui)      │          │  (FastAPI)         │
   │                 │          │                    │
   │  127.0.0.1:3000 │          │  0.0.0.0:8080      │
   └───────┬─────────┘          └─────────┬──────────┘
           │                              │
           │  OpenAI SDK                  │
           └──────────────┬───────────────┘
                          │
                ┌─────────▼──────────┐
                │  Inference Engine  │
                │  (vLLM)            │
                │                    │
                │  0.0.0.0:8000      │
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │       GPU          │
                └────────────────────┘
```

## Components

### nginx — public surface, port 8888

Single-port reverse proxy. Routes:

| Path | Target | Notes |
|---|---|---|
| `/v1/*` | Sohn proxy | OpenAI-compatible API (Sohn persona auto-injected) |
| `/raw/v1/*` | Sohn proxy | Same engine, no persona |
| `/raw/*` | Sohn proxy | Alias for clients that strip `/v1` (e.g. ElevenLabs) |
| `/sohn-health` | Sohn proxy | Liveness check |
| `/status`, `/status/history.json` | static files | Status page |
| `/` (everything else) | acti-ui | Chat UI |

WebSocket upgrade is enabled on `/`. Streaming buffering is disabled across the board so SSE works for both API streaming and the chat UI.

Config: [`platform/nginx/nginx.conf`](../platform/nginx/nginx.conf).

### Sohn API proxy — port 8080

A FastAPI service that:

1. Enforces Bearer-token auth from a key file
2. **On `/v1/*`**: injects Sohn's system prompt at the top of every chat request — Sohn's identity always wins
3. **On `/raw/v1/*` and `/raw/*`**: forwards verbatim — caller brings its own system prompt (used by ElevenLabs Custom LLM, Athena, agent frameworks)
4. Defaults thinking mode to OFF on raw passthrough (voice agents need fast first-token)
5. Scrubs vendor-specific fields (`prompt_token_ids`, `token_ids`, `stop_reason`) from upstream responses so strict OpenAI clients don't choke
6. Normalizes any model alias the client sends to `Sohn`
7. Maintains a single persistent `httpx` client to the inference engine — no per-request connection setup

Code: [`platform/proxy/sohn_proxy.py`](../platform/proxy/sohn_proxy.py).

### Sohn system prompt

The Sohn persona — Unblinded Results Formula, hard rules, anti-sycophancy clause, few-shot examples, persona stability anchor.

Stored as a plain text file and read at proxy startup. Restarting the proxy reloads the prompt.

File: [`platform/system_prompts/sohn.txt`](../platform/system_prompts/sohn.txt).

### Inference engine — port 8000

vLLM serving an OpenAI-compatible API on `:8000`. Bound to all interfaces but only the proxy talks to it. Configured for:

- Long context (262k tokens)
- Sohn alias as the served model name (so vLLM's own `/v1/models` returns just `Sohn`)
- Tool calling (standard tool-call parser format)
- Reasoning parser (separates chain-of-thought into `reasoning_content`)
- Prefix caching (system prompt hits the cache after the first call)
- TRITON attention backend (best perf on the hybrid model arch we use)
- aiter MHA / MoE / linear / RMSNorm kernels enabled

Launch script: [`platform/inference/launch_sohn.sh`](../platform/inference/launch_sohn.sh).

### ACTi Chat UI — acti-ui

The chat UI lives at `vendor/acti-ui/` (a git submodule against an
upstream-tracked, ACTi-branded fork). Configured to:

- Use the ACTi gateway as its only OpenAI-compatible backend
- Display only `Sohn` in the model picker
- Branded as "ACTi" — branding patches live natively in the fork
- Web search enabled (DuckDuckGo, no API key required)
- Image / Video toggles in the chat `+` button (route through the gateway's
  `/v1/images/generations` and `/v1/videos/generations`)
- Auth on, first signup → admin
- Skills / Memory / Knowledge surfaces exposed under Workspace

Launch: [`platform/ui/launch_owui.sh`](../platform/ui/launch_owui.sh).

### Status page

A self-contained HTML file polled every 10 seconds + a Python collector that probes the health endpoint every 60 seconds and writes a 24-hour rolling JSON file.

The page shows: composite status banner, 24-hour uptime %, last-incident time, per-service status (Sohn, ACTi Platform), and a 24-hour timeline.

Files: [`platform/status/status.html`](../platform/status/status.html), [`platform/status/status_collector.py`](../platform/status/status_collector.py).

## Process supervision

We use `tmux` for service supervision in development / single-machine deployment. Each service runs in its own session:

| Session | Service |
|---|---|
| `acti-inference` | vLLM engine |
| `acti-proxy` | Sohn API proxy |
| `acti-ui` | ACTi chat UI |
| `acti-status` | Status collector |

Restarting any service: `tmux kill-session -t <name>` followed by the corresponding launch script.

For production with auto-restart, swap tmux for `systemd` units. Drop-in unit files are not yet included; if needed, add to `startup/systemd/`.

## State on disk

| Path | What |
|---|---|
| `/opt/acti/` | Symlinks / installed copies of platform files |
| `/var/lib/acti/api-keys.txt` | API keys (one per line; `#` for comments) |
| `/var/lib/acti/openwebui/` | acti-ui SQLite DB, uploads, vector store |
| `/var/lib/acti/hf-cache/` | Model weight cache |
| `/var/log/acti/` | Service logs |
| `/usr/share/nginx/html/acti-status/` | Status page static files (nginx-readable) |

These paths are operator-configurable via environment variables. See [`startup/env.example`](../startup/env.example).

## Networking

| Port | Bind | Purpose |
|---|---|---|
| 8888 | `0.0.0.0` | nginx — sole public-facing port |
| 8080 | `0.0.0.0` | Sohn API proxy (also localhost-fronted via nginx) |
| 8000 | `0.0.0.0` | vLLM (internal use only — should not be in any firewall allowlist beyond localhost) |
| 3000 | `127.0.0.1` | acti-ui (internal) |

## Behavioral evaluation

A self-contained eval framework lives at [`platform/eval/`](../platform/eval/). It calls `/raw/v1` (no persona injection — does not write to cortex memory or trigger skill-sync) and writes only to `/opt/acti/eval/{cache,runs}`. Used as a checkpoint gate before any change to `sohn.txt`, the proxy's tool wiring, or the retrieval service. See [`EVAL.md`](EVAL.md) for the rubric, the iteration history with metrics, cross-validation results, and the bug-fix write-ups.

## Failure modes & recovery

| Symptom | Likely cause | Recovery |
|---|---|---|
| `/sohn-health` returns `{"status":"down"}` | vLLM crashed or still loading | `tmux attach -t acti-inference`, check log |
| `/v1/*` 502 | proxy can reach nginx but not vLLM | restart proxy: `tmux kill-session -t acti-proxy && bash platform/proxy/launch.sh` |
| Status page 403 | nginx user can't read static path | `chown -R www-data:www-data /usr/share/nginx/html/acti-status/` |
| chat UI shows duplicate users | `DEFAULT_USER_ROLE` set during onboarding (don't do that) | dedupe + promote first user; see ops notes |
| ElevenLabs `custom_llm_error: failed to generate` | thinking mode emitting reasoning tokens before content | use `/raw/v1` (already defaults thinking off) |
