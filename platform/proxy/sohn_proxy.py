"""
Sohn API — OpenAI-compatible proxy over vLLM.

Endpoints:
  POST /v1/chat/completions   — chat (streaming + tools)
  POST /v1/completions        — legacy completion, routed through chat (keeps Sohn identity)
  GET  /v1/models             — advertises only "Sohn"
  GET  /health                — upstream health

Auth:
  Optional. If the configured API keys file exists with >=1 non-blank line,
  every request must carry `Authorization: Bearer <key>`. If no file (or empty),
  auth is disabled — useful in dev.

CORS:
  Enabled for any origin so browser apps can call it.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import os
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


# ---------- configuration ----------

SYSTEM_PROMPT = Path(os.environ.get("ACTI_SYSTEM_PROMPT_PATH", "/opt/acti/system_prompts/sohn.txt")).read_text()
VLLM_URL = "http://127.0.0.1:8000"
SERVED_NAME = "Sohn"
API_KEYS_FILE = Path(os.environ.get("ACTI_API_KEY_FILE", "/var/lib/acti/api-keys.txt"))


def load_api_keys() -> set[str]:
    if not API_KEYS_FILE.exists():
        return set()
    keys = {
        line.strip()
        for line in API_KEYS_FILE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return keys


# ---------- app lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(
        max_connections=100, max_keepalive_connections=20, keepalive_expiry=60.0
    )
    app.state.client = httpx.AsyncClient(base_url=VLLM_URL, timeout=None, limits=limits)
    app.state.api_keys = load_api_keys()
    print(
        f"[sohn-proxy] started. auth={'ON' if app.state.api_keys else 'OFF (dev)'} "
        f"keys_loaded={len(app.state.api_keys)}"
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="Sohn API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- helpers ----------

def _require_auth(request: Request) -> None:
    keys = request.app.state.api_keys
    if not keys:
        return  # dev mode
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing Authorization: Bearer <api_key>",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )
    token = header[7:].strip()
    if token not in keys:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )


def _inject_system_prompt(messages: list) -> list:
    if messages and messages[0].get("role") == "system":
        user_system = messages[0].get("content", "")
        if isinstance(user_system, list):
            user_system = "\n".join(
                part.get("text", "") for part in user_system if isinstance(part, dict)
            )
        merged = (
            SYSTEM_PROMPT
            + "\n\n---\n\n## Additional Context From User\n\n"
            + str(user_system)
        )
        return [{"role": "system", "content": merged}] + messages[1:]
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages


def _error(status: int, message: str, etype: str = "server_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype}},
    )


# ---------- endpoints ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _require_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        return _error(400, f"Invalid JSON body: {e}", "invalid_request_error")

    body["messages"] = _inject_system_prompt(body.get("messages", []))
    body["model"] = SERVED_NAME
    stream = bool(body.get("stream"))

    client: httpx.AsyncClient = request.app.state.client

    if stream:
        async def event_stream():
            try:
                async with client.stream(
                    "POST", "/v1/chat/completions", json=body
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except httpx.HTTPError as e:
                yield f"data: {{\"error\":{{\"message\":\"{e}\",\"type\":\"upstream_error\"}}}}\n\n".encode()
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        resp = await client.post("/v1/chat/completions", json=body)
    except httpx.HTTPError as e:
        return _error(502, f"Upstream error: {e}", "upstream_error")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/completions")
async def completions(request: Request):
    """Legacy completions endpoint — internally routed through chat so Sohn identity is preserved."""
    _require_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        return _error(400, f"Invalid JSON body: {e}", "invalid_request_error")

    prompt = body.get("prompt")
    if isinstance(prompt, list):
        prompt = "\n\n".join(str(p) for p in prompt)
    if not isinstance(prompt, str) or not prompt:
        return _error(400, "`prompt` must be a non-empty string", "invalid_request_error")

    chat_body = {
        "model": SERVED_NAME,
        "messages": _inject_system_prompt([{"role": "user", "content": prompt}]),
        "max_tokens": body.get("max_tokens", 512),
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p", 0.95),
        "stream": bool(body.get("stream")),
    }
    for opt in (
        "presence_penalty", "frequency_penalty", "stop", "seed", "logprobs",
        "top_k", "min_p", "repetition_penalty", "chat_template_kwargs",
    ):
        if opt in body:
            chat_body[opt] = body[opt]

    client: httpx.AsyncClient = request.app.state.client

    if chat_body["stream"]:
        async def event_stream():
            async with client.stream("POST", "/v1/chat/completions", json=chat_body) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        resp = await client.post("/v1/chat/completions", json=chat_body)
    except httpx.HTTPError as e:
        return _error(502, f"Upstream error: {e}", "upstream_error")
    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    chat = resp.json()
    choice = chat["choices"][0]
    msg = choice.get("message", {})
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return {
        "id": "cmpl-" + uuid.uuid4().hex[:24],
        "object": "text_completion",
        "created": int(time.time()),
        "model": SERVED_NAME,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": choice.get("finish_reason", "stop"),
            }
        ],
        "usage": chat.get("usage", {}),
    }


@app.get("/v1/models")
async def list_models(request: Request):
    _require_auth(request)
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "unblinded",
            }
        ],
    }


# ============ RAW PASSTHROUGH ENDPOINTS (no Sohn system-prompt injection) ============
# Use these when the caller brings its own identity / system prompt — e.g. ElevenLabs
# Custom LLM with an Athena persona. Auth is still enforced. The client's `messages`
# array is forwarded verbatim; the model name is normalized to what vLLM serves.

_VLLM_EXTRA_FIELDS = {
    "prompt_token_ids", "token_ids", "stop_reason",
    "prompt_logprobs", "kv_transfer_params",
}


def _scrub_openai(obj):
    """Recursively strip non-OpenAI fields some strict clients (ElevenLabs) reject."""
    if isinstance(obj, dict):
        return {k: _scrub_openai(v) for k, v in obj.items() if k not in _VLLM_EXTRA_FIELDS}
    if isinstance(obj, list):
        return [_scrub_openai(v) for v in obj]
    return obj


def _scrub_sse_line(line: bytes) -> bytes:
    """Scrub a single SSE 'data: {...}' line; pass through [DONE] and empty lines."""
    if not line.startswith(b"data:"):
        return line
    payload = line[5:].strip()
    if not payload or payload == b"[DONE]":
        return line
    import json as _json
    try:
        obj = _json.loads(payload)
    except Exception:
        return line
    return b"data: " + _json.dumps(_scrub_openai(obj), separators=(",", ":")).encode() + b"\n"


@app.post("/raw/v1/chat/completions")
async def raw_chat_completions(request: Request):
    _require_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        return _error(400, f"Invalid JSON body: {e}", "invalid_request_error")

    # Normalize model name so any placeholder (e.g. "athena", "gpt-4o") hits our engine.
    body["model"] = SERVED_NAME

    # Default thinking OFF for the passthrough endpoint. Voice agents (ElevenLabs) and
    # latency-sensitive callers don't know about the upstream model's thinking mode, which would
    # otherwise emit 500-2000 reasoning_content tokens before the first content token.
    ctk = body.setdefault("chat_template_kwargs", {})
    ctk.setdefault("enable_thinking", False)

    stream = bool(body.get("stream"))
    client: httpx.AsyncClient = request.app.state.client

    # Debug: log request shape (first call only, no 23k system prompt — truncate).
    import json as _json
    try:
        msgs = body.get("messages", [])
        sys_len = len(msgs[0]["content"]) if msgs and msgs[0].get("role") == "system" else 0
        tools_n = len(body.get("tools", []) or [])
        print(f"[raw/v1] stream={stream} msgs={len(msgs)} sys_chars={sys_len} tools={tools_n} "
              f"max_tokens={body.get('max_tokens')} keys={list(body.keys())}")
    except Exception:
        pass

    if stream:
        async def event_stream():
            buf = b""
            async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        scrubbed = _scrub_sse_line(line)
                        yield scrubbed + b"\n"
                if buf:
                    yield _scrub_sse_line(buf)
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        resp = await client.post("/v1/chat/completions", json=body)
    except httpx.HTTPError as e:
        return _error(502, f"Upstream error: {e}", "upstream_error")
    return JSONResponse(content=_scrub_openai(resp.json()), status_code=resp.status_code)


@app.post("/raw/v1/completions")
async def raw_completions(request: Request):
    _require_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        return _error(400, f"Invalid JSON body: {e}", "invalid_request_error")
    body["model"] = SERVED_NAME
    client: httpx.AsyncClient = request.app.state.client
    try:
        resp = await client.post("/v1/completions", json=body)
    except httpx.HTTPError as e:
        return _error(502, f"Upstream error: {e}", "upstream_error")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/raw/v1/models")
async def raw_list_models(request: Request):
    """Returns the underlying model as a generic OpenAI-style entry — no Sohn branding.
    ElevenLabs will offer this name in its model dropdown."""
    _require_auth(request)
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "custom",
            }
        ],
    }


# --- Aliases WITHOUT /v1 — some clients (e.g. ElevenLabs) strip /v1 from base URL ---
app.add_api_route("/raw/chat/completions", raw_chat_completions, methods=["POST"])
app.add_api_route("/raw/completions", raw_completions, methods=["POST"])
app.add_api_route("/raw/models", raw_list_models, methods=["GET"])
# ====================================================================================


@app.get("/health")
async def health(request: Request):
    client: httpx.AsyncClient = request.app.state.client
    try:
        r = await client.get("/health", timeout=5.0)
        return {
            "status": "ok" if r.status_code == 200 else "degraded",
            "model": SERVED_NAME,
            "version": "0.0.1",
        }
    except Exception:
        return {"status": "down", "model": SERVED_NAME, "version": "0.0.1"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("SOHN_PROXY_PORT", 8080)),
        log_level="info",
    )
