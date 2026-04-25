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
import json
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
SKILLS_DIR = Path(os.environ.get("ACTI_SKILLS_DIR", "/opt/acti/skills"))
MAX_SKILL_LOADS = int(os.environ.get("ACTI_MAX_SKILL_LOADS", "4"))


def load_api_keys() -> set[str]:
    if not API_KEYS_FILE.exists():
        return set()
    keys = {
        line.strip()
        for line in API_KEYS_FILE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return keys


# ---------- skill system ----------

LOAD_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": (
            "Load the full instructions of a skill from Sohn's skill library. "
            "Call this when the user's request matches a skill listed in <available_skills>. "
            "The skill body is returned as the tool result; follow it for the rest of this turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The skill name, e.g. design-md."},
            },
            "required": ["name"],
        },
    },
}


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal `---`-fenced front matter parser. Returns (meta_dict, body)."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    block, body = text[4:end], text[end + 5:]
    meta: dict = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body


def load_skills() -> dict[str, dict]:
    """Discover SKILL.md files under SKILLS_DIR. Returns {name: {description, body, path}}."""
    skills: dict[str, dict] = {}
    if not SKILLS_DIR.is_dir():
        return skills
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        try:
            meta, body = _parse_frontmatter(skill_md.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[sohn-proxy] skill {skill_md} unreadable: {e}")
            continue
        name = meta.get("name")
        if not name:
            print(f"[sohn-proxy] skill {skill_md} has no `name:` in front matter — skipped")
            continue
        skills[name] = {
            "name": name,
            "description": meta.get("description", ""),
            "body": body.strip(),
            "path": str(skill_md),
        }
    return skills


def _skills_dir_mtime() -> float:
    """Latest mtime across SKILLS_DIR and every SKILL.md under it. Used for hot-reload."""
    if not SKILLS_DIR.is_dir():
        return 0.0
    try:
        latest = SKILLS_DIR.stat().st_mtime
    except FileNotFoundError:
        return 0.0
    for p in SKILLS_DIR.glob("*/SKILL.md"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def _maybe_reload_skills(state) -> None:
    """Re-scan SKILLS_DIR if any SKILL.md changed since the last load. Cheap (one stat per file)."""
    latest = _skills_dir_mtime()
    if latest > getattr(state, "skills_mtime", 0.0):
        state.skills = load_skills()
        state.skills_manifest = _skills_manifest_block(state.skills)
        state.skills_mtime = latest
        print(f"[sohn-proxy] skills hot-reloaded: {list(state.skills.keys()) or 'none'}", flush=True)


def _skills_manifest_block(skills: dict[str, dict]) -> str:
    """Short manifest injected into the system prompt for skill discovery."""
    if not skills:
        return ""
    lines = [
        "<available_skills>",
        "Sohn has a skill library. Each entry below names a skill and its activation criteria. "
        "Call the `load_skill` tool with the skill name to retrieve its full instructions; the "
        "instructions are appended to the conversation as a tool result and you must follow them "
        "for the remainder of the turn. Activate at most one skill per request, and only when the "
        "description clearly matches the user's intent.",
        "",
    ]
    for name, info in skills.items():
        lines.append(f"- **{name}**: {info['description']}")
    lines.append("</available_skills>")
    return "\n".join(lines)


# ---------- app lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(
        max_connections=100, max_keepalive_connections=20, keepalive_expiry=60.0
    )
    app.state.client = httpx.AsyncClient(base_url=VLLM_URL, timeout=None, limits=limits)
    app.state.api_keys = load_api_keys()
    app.state.skills = load_skills()
    app.state.skills_manifest = _skills_manifest_block(app.state.skills)
    app.state.skills_mtime = _skills_dir_mtime()
    print(
        f"[sohn-proxy] started. auth={'ON' if app.state.api_keys else 'OFF (dev)'} "
        f"keys_loaded={len(app.state.api_keys)} "
        f"skills_loaded={list(app.state.skills.keys()) or 'none'}"
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


# ---------- endpoints ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _require_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        return _error(400, f"Invalid JSON body: {e}", "invalid_request_error")

    body["model"] = SERVED_NAME
    stream = bool(body.get("stream"))
    _maybe_reload_skills(request.app.state)
    skills: dict = request.app.state.skills
    client_passed_tools = bool(body.get("tools"))

    # Skill system activates only when:
    #   - the proxy has skills loaded, AND
    #   - the client did not bring its own tools (we don't want to mix our internal
    #     `load_skill` with caller-defined tools — that would force the caller to
    #     execute a tool they did not register), AND
    #   - the request is non-streaming (the streaming code path is left untouched
    #     for backwards compatibility; tool-loop handling under SSE is harder).
    use_skills = bool(skills) and not client_passed_tools and not stream

    body["messages"] = _inject_system_prompt(
        body.get("messages", []),
        skills_manifest=request.app.state.skills_manifest if use_skills else "",
    )

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

    if use_skills:
        body["tools"] = [LOAD_SKILL_TOOL]
        body.setdefault("tool_choice", "auto")
        return await _run_with_skill_loop(client, body, skills)

    try:
        resp = await client.post("/v1/chat/completions", json=body)
    except httpx.HTTPError as e:
        return _error(502, f"Upstream error: {e}", "upstream_error")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def _run_with_skill_loop(
    client: httpx.AsyncClient, body: dict, skills: dict[str, dict]
) -> JSONResponse:
    """Run the inference engine in a loop, executing `load_skill` tool calls in-process.

    Stops as soon as the model returns a turn that contains no `load_skill` tool call,
    or after MAX_SKILL_LOADS rounds (whichever is first). Strips our injected tool
    plumbing out of the final response shape so callers do not see it.
    """
    last_data: dict | None = None
    for _ in range(MAX_SKILL_LOADS + 1):
        try:
            resp = await client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as e:
            return _error(502, f"Upstream error: {e}", "upstream_error")
        if resp.status_code != 200:
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

        data = resp.json()
        last_data = data
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []
        load_calls = [
            t for t in tool_calls
            if (t.get("function") or {}).get("name") == "load_skill"
        ]
        if not load_calls:
            # Either no tool calls at all, or the model is asking for a non-skill
            # tool — but we never advertised any other tools, so this should not
            # happen. Either way, return as-is.
            return JSONResponse(content=data, status_code=200)

        # Append the assistant's tool-call message and a tool-result for each load.
        body["messages"].append(msg)
        for tc in load_calls:
            try:
                args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            sname = args.get("name", "")
            sk = skills.get(sname)
            if sk:
                content = sk["body"]
            else:
                content = (
                    f"ERROR: skill '{sname}' is not available. "
                    f"Known skills: {sorted(skills.keys())}. "
                    f"Continue without loading any skill."
                )
            body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": "load_skill",
                "content": content,
            })

    # Iteration cap reached — return whatever the model last produced.
    return JSONResponse(content=last_data or {}, status_code=200)


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
