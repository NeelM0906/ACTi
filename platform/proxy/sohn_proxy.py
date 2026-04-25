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
import asyncio
import hashlib
import json
import mimetypes
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
# Caps the number of model→tool round trips per chat turn. Same constant
# bounds the legacy non-streaming skill loop AND the streaming agentic
# loop. Env-var name kept for backwards compat; semantically it's "max
# agent turns".
MAX_SKILL_LOADS = int(os.environ.get("ACTI_MAX_SKILL_LOADS", "4"))
AGENT_KEEPALIVE_INTERVAL_S = float(os.environ.get("ACTI_AGENT_KEEPALIVE_INTERVAL_S", "15"))

# ---------- Lumen (image / video generation) configuration ----------
# Image and video generation are delegated to the Lumen FastAPI backend
# running on a separate workstation, exposed through ngrok. The ACTi proxy
# never hands the Lumen token to the model or the client — it lives only
# in the proxy process environment.
LUMEN_BASE_URL = os.environ.get("ACTI_LUMEN_BASE_URL", "").rstrip("/")
LUMEN_AUTH_TOKEN = os.environ.get("ACTI_LUMEN_AUTH_TOKEN", "")
MEDIA_DIR = Path(os.environ.get("ACTI_MEDIA_DIR", "/var/lib/acti/media"))
LUMEN_IMAGE_TIMEOUT = float(os.environ.get("ACTI_LUMEN_IMAGE_TIMEOUT", "180"))
LUMEN_VIDEO_TIMEOUT = float(os.environ.get("ACTI_LUMEN_VIDEO_TIMEOUT", "600"))


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

GENERATE_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Create one or more original images from a text description. "
            "Call this only when the user explicitly asks for an image to be drawn, generated, "
            "rendered, or visualised. Do NOT call it to fetch existing images from the web — "
            "this tool always synthesises new pixels. Generation typically takes 5–20 seconds. "
            "The tool result is a list of URLs you must reference in your final reply with "
            "standard markdown image syntax: ![short alt text](url)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "What to draw. Be visual and concrete (subject, style, lighting, framing).",
                },
                "width": {"type": "integer", "description": "Pixels. Default 1024.", "default": 1024},
                "height": {"type": "integer", "description": "Pixels. Default 1024.", "default": 1024},
                "num_images": {
                    "type": "integer",
                    "description": "How many variants to generate. 1–4. Default 1.",
                    "default": 1,
                },
            },
            "required": ["prompt"],
        },
    },
}

GENERATE_VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_video",
        "description": (
            "Create a short original video clip from a text description. "
            "Call this only when the user explicitly asks for a video / clip / animation / movie. "
            "Generation takes 30–180 seconds — use sparingly. The tool result is a single URL "
            "you must embed in your final reply with an HTML5 video tag: "
            "<video controls src=\"...\"></video>."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "What the clip shows. Describe action, camera, mood, and setting.",
                },
                "duration": {
                    "type": "integer",
                    "description": "Length in seconds. 1–8. Default 4.",
                    "default": 4,
                },
                "resolution": {
                    "type": "string",
                    "enum": ["540p", "720p", "1080p"],
                    "description": "Output resolution. Default 1080p.",
                    "default": "1080p",
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["16:9", "9:16"],
                    "description": "Frame aspect ratio. Default 16:9.",
                    "default": "16:9",
                },
                "camera_motion": {
                    "type": "string",
                    "enum": ["none", "dolly_in", "dolly_out", "dolly_left", "dolly_right",
                             "jib_up", "jib_down", "static", "focus_shift"],
                    "description": "Optional camera move. Default none.",
                    "default": "none",
                },
            },
            "required": ["prompt"],
        },
    },
}


def media_enabled() -> bool:
    return bool(LUMEN_BASE_URL and LUMEN_AUTH_TOKEN)


# Single-flight lock — Lumen runs on one workstation GPU and serialises
# generations server-side. Holding the lock proxy-side prevents a second
# caller from blocking inside Lumen's queue with no progress signal.
_lumen_lock = asyncio.Lock()


async def _mirror_lumen_file(client: httpx.AsyncClient, abs_path: str) -> str | None:
    """Fetch a file from Lumen's /api/files, save under MEDIA_DIR, return public URL.

    Returns a relative URL like `/media/<sha>.<ext>` so it works regardless of
    which host the OWUI client used to reach the platform. If the download
    fails (auth, not found, transport error), returns None.
    """
    headers = {"Authorization": f"Bearer {LUMEN_AUTH_TOKEN}"}
    sha = hashlib.sha256(f"{abs_path}|{time.time_ns()}".encode()).hexdigest()[:24]
    ext = Path(abs_path).suffix or ".bin"
    fname = f"{sha}{ext}"
    dest = MEDIA_DIR / fname
    try:
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        async with client.stream(
            "GET",
            f"{LUMEN_BASE_URL}/api/files",
            params={"path": abs_path},
            headers=headers,
        ) as r:
            if r.status_code != 200:
                print(f"[sohn-proxy] lumen file fetch {r.status_code} for {abs_path}", flush=True)
                return None
            with dest.open("wb") as f:
                async for chunk in r.aiter_bytes():
                    f.write(chunk)
    except (httpx.HTTPError, OSError) as e:
        print(f"[sohn-proxy] lumen file mirror failed: {e}", flush=True)
        return None
    return f"/media/{fname}"


async def _lumen_generate_image(args: dict) -> dict:
    """Call Lumen image generation, mirror outputs, return {urls: [...]} or {error}."""
    if not media_enabled():
        return {"error": "image generation is not configured on this deployment."}
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required."}
    payload = {
        "prompt": prompt,
        "width": int(args.get("width") or 1024),
        "height": int(args.get("height") or 1024),
        "numSteps": 4,
        "numImages": max(1, min(int(args.get("num_images") or 1), 4)),
    }
    headers = {"Authorization": f"Bearer {LUMEN_AUTH_TOKEN}"}
    async with _lumen_lock:
        async with httpx.AsyncClient(timeout=LUMEN_IMAGE_TIMEOUT) as c:
            try:
                resp = await c.post(
                    f"{LUMEN_BASE_URL}/api/generate-image", json=payload, headers=headers
                )
            except httpx.HTTPError as e:
                return {"error": f"upstream connection failed: {e}"}
            if resp.status_code != 200:
                return {"error": f"upstream {resp.status_code}: {resp.text[:300]}"}
            data = resp.json()
            if data.get("status") == "cancelled":
                return {"error": "generation cancelled."}
            if data.get("status") != "complete":
                return {"error": f"generation status: {data.get('status', 'unknown')}"}
            paths = data.get("image_paths") or []
            if not paths:
                return {"error": "no image returned by Lumen."}
            urls: list[str] = []
            for p in paths:
                u = await _mirror_lumen_file(c, p)
                if u is None:
                    return {"error": f"failed to download generated image: {p}"}
                urls.append(u)
    return {"urls": urls, "prompt": prompt}


async def _lumen_generate_video(args: dict) -> dict:
    """Call Lumen video generation, mirror output, return {url} or {error}."""
    if not media_enabled():
        return {"error": "video generation is not configured on this deployment."}
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required."}
    payload = {
        "prompt": prompt,
        "resolution": args.get("resolution") or "1080p",
        "model": "fast",
        "cameraMotion": args.get("camera_motion") or "none",
        "negativePrompt": "",
        "duration": max(1, min(int(args.get("duration") or 4), 8)),
        "fps": 24,
        "audio": False,
        "imagePath": None,
        "audioPath": None,
        "aspectRatio": args.get("aspect_ratio") or "16:9",
    }
    headers = {"Authorization": f"Bearer {LUMEN_AUTH_TOKEN}"}
    async with _lumen_lock:
        async with httpx.AsyncClient(timeout=LUMEN_VIDEO_TIMEOUT) as c:
            try:
                resp = await c.post(
                    f"{LUMEN_BASE_URL}/api/generate", json=payload, headers=headers
                )
            except httpx.HTTPError as e:
                return {"error": f"upstream connection failed: {e}"}
            if resp.status_code != 200:
                return {"error": f"upstream {resp.status_code}: {resp.text[:300]}"}
            data = resp.json()
            if data.get("status") == "cancelled":
                return {"error": "generation cancelled."}
            if data.get("status") != "complete":
                return {"error": f"generation status: {data.get('status', 'unknown')}"}
            video_path = data.get("video_path")
            if not video_path:
                return {"error": "no video returned by Lumen."}
            url = await _mirror_lumen_file(c, video_path)
            if url is None:
                return {"error": "failed to download generated video."}
    return {"url": url, "prompt": prompt}


def _format_media_tool_result(name: str, args: dict, result: dict) -> str:
    """Stringify the result of a media tool call for the model to consume.

    The instruction text deliberately omits the literal embed snippet to
    avoid the model parroting it back twice (once from the tool result,
    once from its own reply). We give the URL and a one-line directive;
    the model fills in the markup.
    """
    if "error" in result:
        return f"ERROR: {result['error']}"
    if name == "generate_image":
        urls = result.get("urls", [])
        lines = ["Image generation complete. URLs:"]
        for u in urls:
            lines.append(f"  {u}")
        lines.append("")
        lines.append(
            "In your reply, write a one-sentence caption and embed each URL "
            "EXACTLY ONCE using markdown image syntax. Do not repeat any URL. "
            "Do not call generate_image again."
        )
        return "\n".join(lines)
    if name == "generate_video":
        url = result.get("url", "")
        return (
            f"Video generation complete. URL:\n  {url}\n\n"
            f"In your reply, write a one-sentence caption and embed the URL "
            f"EXACTLY ONCE using an HTML5 video tag with a src attribute. "
            f"Do not write the tag more than once. Do not call generate_video again."
        )
    return json.dumps(result)


async def _execute_media_tool(name: str, args: dict) -> tuple[str, dict]:
    """Dispatch a media tool call. Returns (tool_result_text, raw_result)."""
    if name == "generate_image":
        result = await _lumen_generate_image(args)
    elif name == "generate_video":
        result = await _lumen_generate_video(args)
    else:
        result = {"error": f"unknown media tool: {name}"}
    return _format_media_tool_result(name, args, result), result


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
        f"skills_loaded={list(app.state.skills.keys()) or 'none'} "
        f"media={'ON ('+LUMEN_BASE_URL+')' if media_enabled() else 'OFF'}"
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

    # Skill system activates when the proxy has skills loaded AND the caller
    # did not bring its own tools. Streaming is supported via an SSE tap that
    # decides between forwarding content deltas verbatim and intercepting a
    # `load_skill` tool call (see _stream_with_skill_loop).
    use_skills = bool(skills) and not client_passed_tools

    media_on = media_enabled() and not client_passed_tools
    use_proxy_tools = use_skills or media_on

    body["messages"] = _inject_system_prompt(
        body.get("messages", []),
        skills_manifest=request.app.state.skills_manifest if use_skills else "",
    )

    client: httpx.AsyncClient = request.app.state.client

    if use_proxy_tools:
        proxy_tools: list[dict] = []
        if use_skills:
            proxy_tools.append(LOAD_SKILL_TOOL)
        if media_on:
            proxy_tools.extend([GENERATE_IMAGE_TOOL, GENERATE_VIDEO_TOOL])
        body["tools"] = proxy_tools
        body.setdefault("tool_choice", "auto")
        if stream:
            return StreamingResponse(
                _agentic_stream(client, body, skills),
                media_type="text/event-stream",
            )
        return await _run_with_skill_loop(client, body, skills)

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


PROXY_TOOL_NAMES = frozenset({"load_skill", "generate_image", "generate_video"})


def _make_content_chunk(text: str) -> bytes:
    """Build an OpenAI-shaped SSE chunk that injects assistant content text.

    Used for synthetic chunks the proxy emits itself (status messages,
    progress narration, error notices) — distinct from passthrough chunks
    that come straight from the engine.
    """
    evt = {
        "id": "agentic-" + uuid.uuid4().hex[:12],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": SERVED_NAME,
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": None,
        }],
    }
    return b"data: " + json.dumps(evt).encode() + b"\n\n"


async def _periodic_progress(queue: asyncio.Queue, label: str) -> None:
    """Emit a small status content chunk every AGENT_KEEPALIVE_INTERVAL_S
    seconds while a tool is running. Doubles as SSE keepalive — content
    chunks keep the SSE connection alive AND show the user something is
    happening.
    """
    elapsed = 0
    try:
        while True:
            await asyncio.sleep(AGENT_KEEPALIVE_INTERVAL_S)
            elapsed += int(AGENT_KEEPALIVE_INTERVAL_S)
            await queue.put(_make_content_chunk(f"_…{label} ({elapsed}s)…_\n"))
    except asyncio.CancelledError:
        return


async def _execute_proxy_tool(tname: str, args: dict, skills: dict) -> str:
    """Dispatch a proxy-handled tool call to its executor. Returns the
    string content for the tool result message.
    """
    if tname == "load_skill":
        sname = args.get("name", "")
        sk = skills.get(sname)
        if sk:
            return sk["body"]
        return (
            f"ERROR: skill '{sname}' is not in the library. "
            f"Known: {sorted(skills.keys())}. Continue without it."
        )
    if tname in {"generate_image", "generate_video"}:
        content, _raw = await _execute_media_tool(tname, args)
        return content
    return f"ERROR: unknown proxy tool '{tname}'."


async def _stream_one_turn(
    client: httpx.AsyncClient, body: dict, queue: asyncio.Queue
) -> list[dict]:
    """Stream one engine turn into `queue`. Return any accumulated tool calls.

    Wire format (OpenAI-compatible, as emitted by SGLang / vLLM):
        data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\\n\\n
        data: {"choices":[{"delta":{"tool_calls":[
            {"index":0,"id":"...","function":{"name":"...","arguments":""}}
        ]}}]}\\n\\n
        data: {"choices":[{"delta":{"tool_calls":[
            {"index":0,"function":{"arguments":"par"}}
        ]}}]}\\n\\n
        ...
        data: {"choices":[{"finish_reason":"tool_calls"}]}\\n\\n
        data: [DONE]\\n\\n

    Per event:
      - `delta.tool_calls` present → accumulate by index, do NOT forward
        (we will execute the tool in the runner; the client should not
        see partial tool_call deltas)
      - `delta.content` or `delta.reasoning_content` → forward verbatim
      - `finish_reason="tool_calls"` → stop, return accumulated calls
      - `finish_reason="stop"` (or any other) → forward, return []
      - `[DONE]` → engine signals end; do not forward (runner emits the
        outer DONE)

    KEY GOTCHA (LiteLLM #20711, OpenAI streaming spec): only the FIRST
    delta per tool call carries `id` and `function.name`; subsequent
    deltas carry only `index` and partial `function.arguments`. Always
    accumulate by `index`.
    """
    stream_body = dict(body)
    stream_body["stream"] = True

    tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None

    try:
        async with client.stream(
            "POST", "/v1/chat/completions", json=stream_body
        ) as resp:
            if resp.status_code != 200:
                content = await resp.aread()
                await queue.put(b"data: " + content + b"\n\n")
                return []

            buf = b""
            async for chunk in resp.aiter_bytes():
                buf += chunk
                while b"\n\n" in buf:
                    raw_event, _, buf = buf.partition(b"\n\n")
                    line = raw_event.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        # Engine done; runner emits outer DONE.
                        if finish_reason is None:
                            finish_reason = "stop"
                        continue
                    try:
                        evt = json.loads(payload)
                    except Exception:
                        # Unparseable: forward verbatim.
                        await queue.put(b"data: " + payload + b"\n\n")
                        continue

                    choices = evt.get("choices") or []
                    if not choices:
                        # Usage-only chunks, etc — forward.
                        await queue.put(b"data: " + payload + b"\n\n")
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    fr = choice.get("finish_reason")
                    delta_tools = delta.get("tool_calls")

                    if delta_tools:
                        # Accumulate; do not forward the partial tool_call delta.
                        for tcd in delta_tools:
                            idx = tcd.get("index", 0)
                            slot = tool_calls.setdefault(idx, {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                            if tcd.get("id"):
                                slot["id"] = tcd["id"]
                            if tcd.get("type"):
                                slot["type"] = tcd["type"]
                            fnd = tcd.get("function") or {}
                            if fnd.get("name"):
                                slot["function"]["name"] = fnd["name"]
                            if fnd.get("arguments") is not None:
                                slot["function"]["arguments"] += fnd["arguments"]
                    elif (
                        delta.get("content") is not None
                        or delta.get("reasoning_content") is not None
                        or "role" in delta
                    ):
                        # Passthrough content / reasoning / role-init chunks.
                        await queue.put(b"data: " + payload + b"\n\n")
                    elif fr is not None and fr != "tool_calls":
                        # Final stop chunk — forward so client sees finish_reason.
                        await queue.put(b"data: " + payload + b"\n\n")
                    # else: tool_calls finish chunk, drop it

                    if fr is not None:
                        finish_reason = fr
    except httpx.HTTPError as e:
        err = json.dumps({"error": {"message": str(e), "type": "upstream_error"}})
        await queue.put(b"data: " + err.encode() + b"\n\n")
        return []

    if finish_reason == "tool_calls" and tool_calls:
        return [tool_calls[i] for i in sorted(tool_calls.keys())]
    return []


async def _agentic_stream(
    client: httpx.AsyncClient, body: dict, skills: dict[str, dict]
):
    """Single streaming agent loop. Producer/consumer architecture:

      - Background `runner` task drives the engine, executes tools, and
        pushes SSE chunks into an asyncio.Queue.
      - This generator (consumer) yields chunks from the queue to the
        client.

    The loop runs at most MAX_SKILL_LOADS turns. Each turn:
      1. Stream the engine response to the client; passthrough content
         and reasoning deltas verbatim, accumulate any tool_call deltas.
      2. If the turn ended with a tool call, append the assistant's
         tool_call message + execute the tool + append the tool result.
      3. Loop. Continue until a turn ends with no tool call.

    Tools (load_skill, generate_image, generate_video) stay registered
    across all turns — never strip `tools=` mid-conversation. That was
    the cause of the prior chat-template XML leak; with tools always
    advertised, the chat template renders tool_call / tool messages
    correctly.

    Inspired by openai-agents-python's run_streamed pattern (background
    task + queue + RawResponseEvent passthrough).
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def runner() -> None:
        try:
            for turn in range(MAX_SKILL_LOADS + 1):
                t0 = time.time()
                tool_calls = await _stream_one_turn(client, body, queue)
                dt = time.time() - t0
                if not tool_calls:
                    print(f"[agent] turn {turn+1}: stream done in {dt:.1f}s, no tool calls", flush=True)
                    return
                names = ",".join(
                    (tc.get("function") or {}).get("name", "?") for tc in tool_calls
                )
                print(
                    f"[agent] turn {turn+1}: stream done in {dt:.1f}s, "
                    f"tool_calls=[{names}]",
                    flush=True,
                )

                # Append the assistant's tool-call message so subsequent
                # turns see it in chat history.
                body["messages"].append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                })

                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    tname = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    if tname not in PROXY_TOOL_NAMES:
                        body["messages"].append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "name": tname,
                            "content": f"ERROR: unknown tool '{tname}'.",
                        })
                        continue

                    label = {
                        "load_skill": "Loading skill",
                        "generate_image": "Generating image",
                        "generate_video": "Generating video — this can take 30-180s",
                    }.get(tname, f"Calling {tname}")
                    await queue.put(_make_content_chunk(f"\n\n_{label}…_\n\n"))
                    progress_label = {
                        "load_skill": "loading",
                        "generate_image": "still generating",
                        "generate_video": "still generating",
                    }.get(tname, "still working")

                    progress_task = asyncio.create_task(
                        _periodic_progress(queue, progress_label)
                    )
                    t_tool = time.time()
                    try:
                        content = await _execute_proxy_tool(tname, args, skills)
                    except Exception as e:
                        content = f"ERROR: tool execution raised: {e}"
                    finally:
                        progress_task.cancel()
                        try:
                            await progress_task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001
                            pass
                    print(
                        f"[agent] turn {turn+1}: {tname} took "
                        f"{time.time()-t_tool:.1f}s, result_len={len(content)}",
                        flush=True,
                    )

                    body["messages"].append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": tname,
                        "content": content,
                    })
            else:
                print(f"[agent] hit MAX_SKILL_LOADS={MAX_SKILL_LOADS}", flush=True)
                await queue.put(_make_content_chunk(
                    "\n\n_Reached the per-turn tool-call limit. Stopping here._\n"
                ))
        except Exception as e:  # noqa: BLE001
            print(f"[agent] runner unexpected error: {e}", flush=True)
            err = json.dumps({"error": {"message": str(e), "type": "agent_error"}})
            await queue.put(b"data: " + err.encode() + b"\n\n")
        finally:
            await queue.put(b"data: [DONE]\n\n")
            await queue.put(None)  # sentinel

    task = asyncio.create_task(runner())
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            yield chunk
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


async def _run_with_skill_loop(
    client: httpx.AsyncClient, body: dict, skills: dict[str, dict]
) -> JSONResponse:
    """Non-streamed equivalent of _stream_with_skill_loop.

    Loops the inference engine, executing `load_skill` / `generate_image` /
    `generate_video` tool calls in-process. Returns when the model produces
    a turn with no proxy-handled tool calls, or after MAX_SKILL_LOADS rounds.
    """
    PROXY_TOOL_NAMES = {"load_skill", "generate_image", "generate_video"}
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
        proxy_calls = [
            t for t in tool_calls
            if (t.get("function") or {}).get("name") in PROXY_TOOL_NAMES
        ]
        if not proxy_calls:
            return JSONResponse(content=data, status_code=200)

        body["messages"].append(msg)
        for tc in proxy_calls:
            fn = tc.get("function") or {}
            tname = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            if tname == "load_skill":
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
            elif tname in {"generate_image", "generate_video"}:
                content, _raw = await _execute_media_tool(tname, args)
            else:
                content = f"ERROR: unknown tool '{tname}'."
            body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": tname,
                "content": content,
            })

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


@app.post("/v1/images/generations")
async def images_generations(request: Request):
    """OpenAI-compatible image generation endpoint.

    Mirrors OpenAI's API shape so the OpenAI SDK and other compat clients can
    drop in directly. The actual generation is performed by Lumen; we mirror
    the bytes onto the platform under /media/<sha>.<ext>.
    """
    _require_auth(request)
    if not media_enabled():
        return _error(503, "image generation is not configured on this deployment.", "service_unavailable")
    try:
        body = await request.json()
    except Exception as e:
        return _error(400, f"Invalid JSON body: {e}", "invalid_request_error")

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return _error(400, "`prompt` is required", "invalid_request_error")

    # Map OpenAI-shaped fields to Lumen
    n = max(1, min(int(body.get("n", 1)), 4))
    width, height = 1024, 1024
    size = body.get("size")
    if isinstance(size, str) and "x" in size:
        try:
            w, h = size.split("x", 1)
            width, height = int(w), int(h)
        except ValueError:
            pass

    result = await _lumen_generate_image({
        "prompt": prompt, "width": width, "height": height, "num_images": n,
    })
    if "error" in result:
        return _error(502, result["error"], "upstream_error")

    return {
        "created": int(time.time()),
        "data": [{"url": u, "revised_prompt": prompt} for u in result["urls"]],
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
