"""
ACTi gateway — OpenAI-compatible HTTP front door.

Endpoints:
  POST /v1/chat/completions     — chat (streaming + tool dispatch via Spark)
  POST /v1/completions          — legacy completion, routed through chat
  POST /v1/images/generations   — direct image generation (Lumen → /media)
  GET  /v1/models               — advertises only "Sohn"
  POST /raw/v1/chat/completions — passthrough (no Sohn system prompt)
  GET  /health                  — upstream health

The gateway is the HTTP layer only. Agent runtime lives in spark.py;
tool implementations live in skills.py and media.py; memory will live
in cortex.py. This file contains just routing, auth, system-prompt
mounting, and the wiring glue that registers tools with Spark.

Auth:
  Optional. If the configured API keys file exists with >=1 non-blank line,
  every request must carry `Authorization: Bearer <key>`. If no file (or
  empty), auth is disabled — useful in dev.

CORS:
  Enabled for any origin so browser apps can call it.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import cortex
import media
import skills
from spark import ToolLabels, run_agent_stream, run_agent_sync


# ---------- configuration ----------

SYSTEM_PROMPT = Path(
    os.environ.get("ACTI_SYSTEM_PROMPT_PATH", "/opt/acti/system_prompts/sohn.txt")
).read_text()
ENGINE_URL = "http://127.0.0.1:8000"
SERVED_NAME = "Sohn"
API_KEYS_FILE = Path(os.environ.get("ACTI_API_KEY_FILE", "/var/lib/acti/api-keys.txt"))
MAX_AGENT_TURNS = int(os.environ.get("ACTI_MAX_SKILL_LOADS", "4"))
AGENT_KEEPALIVE_INTERVAL_S = float(os.environ.get("ACTI_AGENT_KEEPALIVE_INTERVAL_S", "15"))

# ---------- Cortex (memory) configuration ----------
MEMORY_DIR = Path(os.environ.get("ACTI_MEMORY_DIR", "/var/lib/acti/memory"))
CONTEXT_WINDOW_TOKENS = int(os.environ.get("ACTI_CONTEXT_WINDOW_TOKENS", "256000"))
COMPACT_BUFFER_TOKENS = int(os.environ.get("ACTI_COMPACT_BUFFER_TOKENS", "16000"))
MEMORY_EXTRACTION_ENABLED = os.environ.get("ACTI_MEMORY_EXTRACTION", "1") == "1"


def _log(msg: str) -> None:
    print(msg, file=sys.stdout, flush=True)


def load_api_keys() -> set[str]:
    if not API_KEYS_FILE.exists():
        return set()
    return {
        line.strip()
        for line in API_KEYS_FILE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


# ---------- Spark wiring ----------

_TOOL_LABELS: dict[str, ToolLabels] = {
    "load_skill":     ToolLabels(start="Loading skill",
                                 progress="loading"),
    "generate_image": ToolLabels(start="Generating image",
                                 progress="still generating"),
    "generate_video": ToolLabels(start="Generating video — this can take 30-180s",
                                 progress="still generating"),
}


def _build_tool_handlers(skill_lib: dict[str, dict]) -> dict:
    """Map of {tool_name: async handler(args) -> str} for Spark to dispatch.

    Closes over the live skill library so a hot-reload is visible to
    subsequent handler invocations without rewiring.
    """
    async def _load_skill(args: dict) -> str:
        return await skills.handle_load_skill(args, skill_lib)

    return {
        "load_skill":     _load_skill,
        "generate_image": media.handle_generate_image,
        "generate_video": media.handle_generate_video,
    }


# ---------- app lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(
        max_connections=100, max_keepalive_connections=20, keepalive_expiry=60.0
    )
    app.state.client = httpx.AsyncClient(base_url=ENGINE_URL, timeout=None, limits=limits)
    app.state.api_keys = load_api_keys()
    app.state.skills = skills.load_skills()
    app.state.skills_manifest = skills.skills_manifest_block(app.state.skills)
    app.state.skills_mtime = skills.skills_dir_mtime()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    n_memories = len(list(MEMORY_DIR.glob("*.md"))) - (1 if (MEMORY_DIR / "MEMORY.md").exists() else 0)
    _log(
        f"[gateway] started. auth={'ON' if app.state.api_keys else 'OFF (dev)'} "
        f"keys_loaded={len(app.state.api_keys)} "
        f"skills_loaded={list(app.state.skills.keys()) or 'none'} "
        f"media={'ON ('+media.LUMEN_BASE_URL+')' if media.media_enabled() else 'OFF'} "
        f"memory={MEMORY_DIR} ({n_memories} files, extraction={'ON' if MEMORY_EXTRACTION_ENABLED else 'OFF'})"
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="ACTi gateway", version="1.0", lifespan=lifespan)

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
            detail={"error": {"message": "Missing Authorization: Bearer <api_key>",
                              "type": "invalid_request_error",
                              "code": "invalid_api_key"}},
        )
    token = header[7:].strip()
    if token not in keys:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key",
                              "type": "invalid_request_error",
                              "code": "invalid_api_key"}},
        )


def _inject_system_prompt(messages: list, skills_manifest: str = "") -> list:
    base = SYSTEM_PROMPT
    if skills_manifest:
        base = base.rstrip() + "\n\n" + skills_manifest
    if messages and messages[0].get("role") == "system":
        user_system = messages[0].get("content", "")
        if isinstance(user_system, list):
            user_system = "\n".join(
                part.get("text", "") for part in user_system if isinstance(part, dict)
            )
        merged = (
            base
            + "\n\n---\n\n## Additional Context From User\n\n"
            + str(user_system)
        )
        return [{"role": "system", "content": merged}] + messages[1:]
    return [{"role": "system", "content": base}] + messages


def _error(status: int, message: str, etype: str = "server_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype}},
    )


# ---------- /v1/chat/completions ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _require_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        return _error(400, f"Invalid JSON body: {e}", "invalid_request_error")

    body["model"] = SERVED_NAME
    stream = bool(body.get("stream"))
    skills.maybe_reload_skills(request.app.state)
    skill_lib: dict = request.app.state.skills
    client_passed_tools = bool(body.get("tools"))

    # Skill / media tools activate when the caller didn't bring its own.
    use_skills = bool(skill_lib) and not client_passed_tools
    use_media = media.media_enabled() and not client_passed_tools
    use_proxy_tools = use_skills or use_media

    body["messages"] = _inject_system_prompt(
        body.get("messages", []),
        skills_manifest=request.app.state.skills_manifest if use_skills else "",
    )

    # Cortex: inject saved memories into the system prompt at request start.
    body["messages"] = cortex.inject_memories(body["messages"], MEMORY_DIR)

    client: httpx.AsyncClient = request.app.state.client

    # Cortex hooks for the agent loop.
    async def _between_turns(msgs: list[dict]) -> list[dict]:
        if cortex.should_compact(
            msgs,
            max_tokens=CONTEXT_WINDOW_TOKENS,
            buffer_tokens=COMPACT_BUFFER_TOKENS,
        ):
            return await cortex.compact(
                client=client, messages=msgs, served_name=SERVED_NAME,
            )
        return msgs

    def _turn_complete(msgs: list[dict]) -> None:
        if not MEMORY_EXTRACTION_ENABLED:
            return
        # Fire and forget — memory extraction must never block the user.
        import asyncio as _aio
        _aio.create_task(cortex.extract_memories(
            client=client, messages=list(msgs), memory_dir=MEMORY_DIR,
            served_name=SERVED_NAME,
        ))

    if use_proxy_tools:
        proxy_tools: list[dict] = []
        if use_skills:
            proxy_tools.append(skills.LOAD_SKILL_TOOL)
        if use_media:
            proxy_tools.extend([media.GENERATE_IMAGE_TOOL, media.GENERATE_VIDEO_TOOL])
        body["tools"] = proxy_tools
        body.setdefault("tool_choice", "auto")

        handlers = _build_tool_handlers(skill_lib)
        if stream:
            return StreamingResponse(
                run_agent_stream(
                    client=client,
                    body=body,
                    tool_handlers=handlers,
                    tool_labels=_TOOL_LABELS,
                    served_name=SERVED_NAME,
                    max_turns=MAX_AGENT_TURNS,
                    keepalive_interval_s=AGENT_KEEPALIVE_INTERVAL_S,
                    on_between_turns=_between_turns,
                    on_turn_complete=_turn_complete,
                ),
                media_type="text/event-stream",
            )
        result = await run_agent_sync(
            client=client, body=body, tool_handlers=handlers, max_turns=MAX_AGENT_TURNS,
            on_between_turns=_between_turns,
            on_turn_complete=_turn_complete,
        )
        if "error" in result:
            return _error(502, result["error"].get("message", "agent error"), "upstream_error")
        return JSONResponse(content=result, status_code=200)

    # Plain pass-through (no proxy-handled tools advertised)
    if stream:
        async def event_stream():
            try:
                async with client.stream(
                    "POST", "/v1/chat/completions", json=body
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except httpx.HTTPError as e:
                yield (b'data: {"error":{"message":"' + str(e).replace('"', "'").encode()
                       + b'","type":"upstream_error"}}\n\n')
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        resp = await client.post("/v1/chat/completions", json=body)
    except httpx.HTTPError as e:
        return _error(502, f"Upstream error: {e}", "upstream_error")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------- /v1/completions ----------

@app.post("/v1/completions")
async def completions(request: Request):
    """Legacy completions endpoint — routed through chat so Sohn identity is preserved."""
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
        "choices": [{
            "text": text, "index": 0, "logprobs": None,
            "finish_reason": choice.get("finish_reason", "stop"),
        }],
        "usage": chat.get("usage", {}),
    }


# ---------- /v1/images/generations ----------

@app.post("/v1/images/generations")
async def images_generations(request: Request):
    """OpenAI-compatible image generation. Backed by Lumen via media.py."""
    _require_auth(request)
    if not media.media_enabled():
        return _error(503, "image generation is not configured on this deployment.",
                      "service_unavailable")
    try:
        body = await request.json()
    except Exception as e:
        return _error(400, f"Invalid JSON body: {e}", "invalid_request_error")

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return _error(400, "`prompt` is required", "invalid_request_error")

    n = max(1, min(int(body.get("n", 1)), 4))
    width, height = 1024, 1024
    size = body.get("size")
    if isinstance(size, str) and "x" in size:
        try:
            w, h = size.split("x", 1)
            width, height = int(w), int(h)
        except ValueError:
            pass

    result = await media.generate_image_raw(prompt, width, height, n)
    if "error" in result:
        return _error(502, result["error"], "upstream_error")
    return {
        "created": int(time.time()),
        "data": [{"url": u, "revised_prompt": prompt} for u in result["urls"]],
    }


# ---------- /v1/models ----------

@app.get("/v1/models")
async def list_models(request: Request):
    _require_auth(request)
    return {
        "object": "list",
        "data": [{"id": SERVED_NAME, "object": "model", "created": 0,
                  "owned_by": "unblinded"}],
    }


# ---------- /raw/* passthrough (no Sohn system-prompt injection) ----------

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
    try:
        obj = json.loads(payload)
    except Exception:
        return line
    return b"data: " + json.dumps(_scrub_openai(obj), separators=(",", ":")).encode() + b"\n"


@app.post("/raw/v1/chat/completions")
async def raw_chat_completions(request: Request):
    _require_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        return _error(400, f"Invalid JSON body: {e}", "invalid_request_error")

    body["model"] = SERVED_NAME
    ctk = body.setdefault("chat_template_kwargs", {})
    ctk.setdefault("enable_thinking", False)
    stream = bool(body.get("stream"))
    client: httpx.AsyncClient = request.app.state.client

    msgs = body.get("messages", [])
    sys_len = len(msgs[0]["content"]) if msgs and msgs[0].get("role") == "system" else 0
    tools_n = len(body.get("tools", []) or [])
    _log(f"[raw/v1] stream={stream} msgs={len(msgs)} sys_chars={sys_len} tools={tools_n} "
         f"max_tokens={body.get('max_tokens')} keys={list(body.keys())}")

    if stream:
        async def event_stream():
            buf = b""
            async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        yield _scrub_sse_line(line) + b"\n"
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
    """Generic OpenAI-style entry — no Sohn branding (for ElevenLabs etc.)."""
    _require_auth(request)
    return {
        "object": "list",
        "data": [{"id": SERVED_NAME, "object": "model", "created": 0, "owned_by": "custom"}],
    }


# Aliases without /v1 — some clients (ElevenLabs) strip /v1 from base URL
app.add_api_route("/raw/chat/completions", raw_chat_completions, methods=["POST"])
app.add_api_route("/raw/completions", raw_completions, methods=["POST"])
app.add_api_route("/raw/models", raw_list_models, methods=["GET"])


# ---------- /health ----------

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
        app, host="0.0.0.0",
        port=int(os.environ.get("SOHN_PROXY_PORT", 8080)),
        log_level="info",
    )
