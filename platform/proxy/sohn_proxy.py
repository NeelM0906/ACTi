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
MAX_SKILL_LOADS = int(os.environ.get("ACTI_MAX_SKILL_LOADS", "4"))

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
    """Stringify the result of a media tool call for the model to consume."""
    if "error" in result:
        return f"ERROR: {result['error']}"
    if name == "generate_image":
        lines = [f"Generated {len(result['urls'])} image(s) for prompt: {result['prompt']!r}."]
        for i, u in enumerate(result["urls"]):
            lines.append(f"- url[{i}]: {u}")
        lines.append(
            "Embed each url in your final reply using markdown image syntax: "
            "![brief alt](url). Do not call generate_image again."
        )
        return "\n".join(lines)
    if name == "generate_video":
        return (
            f"Generated a video for prompt: {result['prompt']!r}.\n"
            f"- url: {result['url']}\n"
            f"Embed the url in your final reply using an HTML5 video tag: "
            f"<video controls src=\"{result['url']}\"></video>. "
            f"Do not call generate_video again."
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
                _stream_with_skill_loop(client, body, skills),
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


async def _stream_with_skill_loop(
    client: httpx.AsyncClient, body: dict, skills: dict[str, dict]
):
    """Two-stage streaming with skill support.

    Stage 1 — non-streamed pre-flight with `enable_thinking=False` and
        `tools=[load_skill]`. The model decides quickly whether to call a
        skill (fast path: tool-call XML emitted in the first few tokens).
        If it does, we read the SKILL.md body off disk and append a tool
        result, then loop. If it doesn't, we exit stage 1 with the user's
        original messages.

    Stage 2 — streamed final response with the user's original
        `chat_template_kwargs` (so thinking comes back if requested), no
        tools, and the augmented messages from stage 1 if a skill loaded.
        RadixAttention serves the prefix cache from stage 1, so stage 2's
        prefill is essentially free.

    Why stage 1 forces thinking off: the upstream model with thinking on
    emits a long stream of `reasoning_content` deltas before deciding
    between content and tool_call. An SSE tap that buffers across that
    reasoning phase causes client-side timeouts (aiohttp's
    TransferEncodingError). Disabling thinking makes the skill-vs-content
    decision happen in the first few output tokens; stage 2 restores
    thinking for the user-visible answer.

    Bounded by MAX_SKILL_LOADS rounds.
    """
    user_ctk = dict(body.get("chat_template_kwargs") or {})
    user_max_tokens = body.get("max_tokens")
    proxy_tools = body.get("tools") or [LOAD_SKILL_TOOL]

    pre_body = dict(body)
    pre_body["messages"] = list(body["messages"])
    pre_body["stream"] = False
    pre_body["tools"] = proxy_tools
    pre_body.setdefault("tool_choice", "auto")
    pre_body["chat_template_kwargs"] = {**user_ctk, "enable_thinking": False}
    # Cap pre-flight: enough to emit a complete tool-call XML (~30 tokens)
    # without paying for a full content reply when no tool is needed.
    pre_body["max_tokens"] = 256

    loaded_skill_blocks: list[str] = []   # skill bodies to inline into stage 2 system prompt
    media_artifacts: list[dict] = []      # [{name, args, urls?, url?, prompt}] for stage 2 inlining

    for _ in range(MAX_SKILL_LOADS + 1):
        try:
            resp = await client.post("/v1/chat/completions", json=pre_body)
        except httpx.HTTPError as e:
            yield (
                b'data: {"error":{"message":"'
                + str(e).replace('"', "'").encode()
                + b'","type":"upstream_error"}}\n\n'
            )
            return
        if resp.status_code != 200:
            yield b"data: " + resp.content + b"\n\n"
            yield b"data: [DONE]\n\n"
            return

        data = resp.json()
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        proxy_calls = [
            tc for tc in tool_calls
            if (tc.get("function") or {}).get("name") in
                {"load_skill", "generate_image", "generate_video"}
        ]
        if not proxy_calls:
            break  # no tools needed — exit pre-flight

        pre_body["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        })
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
                    loaded_skill_blocks.append(f"### Skill: {sname}\n\n{sk['body']}")
                else:
                    content = f"ERROR: skill '{sname}' is not in the library. Continue without."
            elif tname in {"generate_image", "generate_video"}:
                content, raw = await _execute_media_tool(tname, args)
                if "error" not in raw:
                    media_artifacts.append({"name": tname, "args": args, "result": raw})
            else:
                content = f"ERROR: unknown tool '{tname}'."

            pre_body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": tname,
                "content": content,
            })

    # Stage 2 — streamed final response. Inline any loaded skill bodies AND
    # any generated media URLs into the system prompt rather than carrying
    # the tool_call/tool-result round trip into the user-visible turn.
    # Reason: with `tools=` removed, the chat template renders prior tool_calls
    # in messages as raw XML, which the model then parrots back as content.
    # Inlining sidesteps that entirely.
    augmentation_blocks: list[str] = []
    if loaded_skill_blocks:
        augmentation_blocks.append(
            "## Active Skill Instructions\n\n"
            "The following skill(s) have been activated for this turn. Apply their "
            "instructions when producing the user-visible reply.\n\n"
            + "\n\n---\n\n".join(loaded_skill_blocks)
        )
    if media_artifacts:
        media_lines = [
            "## Generated Media For This Reply",
            "",
            "You just generated the following media for the user. Embed them in your reply "
            "exactly as instructed below. Do NOT call generate_image or generate_video again "
            "in this turn — the user already has these results.",
            "",
        ]
        for art in media_artifacts:
            r = art["result"]
            p = r.get("prompt", "")
            if art["name"] == "generate_image":
                for i, u in enumerate(r.get("urls", [])):
                    media_lines.append(
                        f"- Image {i+1} (prompt: {p!r}) — embed with: `![{p[:40]}]({u})`"
                    )
            elif art["name"] == "generate_video":
                u = r.get("url", "")
                media_lines.append(
                    f"- Video (prompt: {p!r}) — embed with: "
                    f"`<video controls src=\"{u}\"></video>`"
                )
        augmentation_blocks.append("\n".join(media_lines))

    if augmentation_blocks:
        original_msgs = body["messages"]
        sys_msg = original_msgs[0] if original_msgs and original_msgs[0].get("role") == "system" else None
        section = "\n\n" + "\n\n".join(augmentation_blocks)
        if sys_msg is not None:
            new_sys = {
                **sys_msg,
                "content": (sys_msg.get("content", "") or "") + section,
            }
            stage2_messages = [new_sys] + original_msgs[1:]
        else:
            stage2_messages = [{"role": "system", "content": section}] + original_msgs
    else:
        stage2_messages = body["messages"]

    final_body = dict(body)
    final_body["messages"] = stage2_messages
    final_body["stream"] = True
    final_body["chat_template_kwargs"] = user_ctk  # restore original thinking setting
    if user_max_tokens is not None:
        final_body["max_tokens"] = user_max_tokens
    final_body.pop("tools", None)
    final_body.pop("tool_choice", None)

    try:
        async with client.stream("POST", "/v1/chat/completions", json=final_body) as resp:
            if resp.status_code != 200:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                return
            async for chunk in resp.aiter_bytes():
                yield chunk
    except httpx.HTTPError as e:
        yield (
            b'data: {"error":{"message":"'
            + str(e).replace('"', "'").encode()
            + b'","type":"upstream_error"}}\n\n'
        )


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
