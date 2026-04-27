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
import re
import sys
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import cortex
import library
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
MAX_AGENT_TURNS = int(os.environ.get("ACTI_MAX_SKILL_LOADS", "6"))
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
    "load_tool":      ToolLabels(start="Activating tool",
                                 progress="activating"),
    "load_skill":     ToolLabels(start="Loading skill",
                                 progress="loading"),
    "generate_image": ToolLabels(start="Generating image",
                                 progress="still generating"),
    "generate_video": ToolLabels(start="Generating video — this can take 30-180s",
                                 progress="still generating"),
    "recall_context": ToolLabels(start="Searching the Unblinded library",
                                 progress="still searching"),
}


# ---------- lazy tool loading ----------
#
# Real tools are NOT pre-registered on body["tools"]. Doing so puts every
# schema into the chat-template's prompt for every request — auxiliary OWUI
# calls (title gen, tag gen, follow-up suggestions) get them too, which:
#   - bloats the prompt (~1500-3000 extra tokens per request)
#   - triggers a known reasoning-suppression behavior in the inference
#     engine: presence of tool schemas pushes the model into a fast
#     tool-call mode with reduced thinking, which interacts unpredictably
#     with the chat-template's thinking block
#   - makes small-budget auxiliary calls deliberate themselves into a
#     content=null / finish_reason=length state
#
# Instead we expose ONE bootstrap tool — `load_tool` — that takes a name and
# registers the requested tool's schema into body["tools"] for use on
# subsequent turns. The model then calls the actual tool normally.
LOAD_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "load_tool",
        "description": (
            "Activate one of the platform's capability tools. Tools are not "
            "pre-registered — call load_tool with the desired tool's name "
            "FIRST, and on your next turn the tool's full schema will be "
            "available for you to call directly with its proper arguments.\n\n"
            "Available tools (call load_tool with one of these names):\n"
            "- load_skill: load a domain skill's full instructions when you "
            "need detailed knowledge or a workflow on a specific topic.\n"
            "- generate_image: produce an image via the Lumen pipeline. "
            "Use when the user asks for an image to be created.\n"
            "- generate_video: produce a short video via the Lumen pipeline. "
            "Use when the user asks for a video to be created.\n"
            "- recall_context: search the Unblinded knowledge corpus for "
            "person dossiers, case files, teaching material, and continuity "
            "snapshots. Use when the user references corpus-specific entities "
            "you don't have in context.\n\n"
            "Do NOT call load_tool for general conversation, simple Q&A, or "
            "anything you can answer from your training and the system prompt. "
            "Only call it when one of the listed capabilities is required."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [
                        "load_skill", "generate_image",
                        "generate_video", "recall_context",
                    ],
                    "description": "The tool to activate.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}


def _real_tool_schemas() -> dict[str, dict]:
    """Map of {tool_name -> full schema} for tools that load_tool can activate."""
    return {
        "load_skill":     skills.LOAD_SKILL_TOOL,
        "generate_image": media.GENERATE_IMAGE_TOOL,
        "generate_video": media.GENERATE_VIDEO_TOOL,
        "recall_context": library.RECALL_CONTEXT_TOOL,
    }


def _build_tool_handlers(skill_lib: dict[str, dict]) -> dict:
    """Map of {tool_name: async handler(args) -> str} for Spark to dispatch.

    Closes over the live skill library so a hot-reload is visible to
    subsequent handler invocations without rewiring.

    Includes handlers for ALL real tools — they're harmless when the
    schema isn't registered on body["tools"] (model can't see them and
    won't call them). When `load_tool` registers a schema mid-conversation,
    the corresponding handler is already present here, ready to dispatch.
    """
    async def _load_skill(args: dict) -> str:
        return await skills.handle_load_skill(args, skill_lib)

    return {
        "load_skill":     _load_skill,
        "generate_image": media.handle_generate_image,
        "generate_video": media.handle_generate_video,
        "recall_context": library.handle_recall_context,
    }


def _make_load_tool_handler(body: dict, allowed: dict[str, dict]):
    """Build a load_tool handler bound to this request's body and allowed
    tool registry. When invoked it appends the requested tool's schema to
    body["tools"] so it's available on subsequent turns."""
    async def _handle(args: dict) -> str:
        name = args.get("name")
        if name not in allowed:
            available = sorted(allowed.keys())
            return (f"ERROR: '{name}' is not an available tool. "
                    f"Available: {available}")
        tools = body.setdefault("tools", [])
        already = any(
            (t.get("function") or {}).get("name") == name for t in tools
        )
        if not already:
            tools.append(allowed[name])
        return (f"Tool '{name}' is now activated. On your next turn, call "
                f"'{name}' directly with the appropriate arguments.")
    return _handle


# ---------- user identity resolution ----------
#
# OWUI forwards the signed-in user's id and full name to the upstream chat
# completion endpoint via HTTP headers when ENABLE_FORWARD_USER_INFO_HEADERS=true.
# We use these to partition cortex memory per user and to seed the user's name
# (collected at signup) so the model addresses them properly from message 1.
#
# Header names match OWUI's defaults. They can be overridden in OWUI's env;
# we keep the defaults here. A header set to an empty string is treated as
# absent.

OWUI_USER_ID_HEADER = "x-openwebui-user-id"
OWUI_USER_NAME_HEADER = "x-openwebui-user-name"


def _resolve_user_identity(request: Request, body: dict) -> tuple[str | None, str | None]:
    """Return (user_id, user_name) for this request.

    Priority for user_id: OWUI header, then body["user"], else None.
    user_name comes from the OWUI header only (URL-decoded). Either may be
    None — the caller decides how to handle that (anonymous partition,
    onboarding block, etc.).
    """
    headers = request.headers
    user_id = (headers.get(OWUI_USER_ID_HEADER) or "").strip() or None
    if not user_id:
        body_user = body.get("user")
        if isinstance(body_user, str) and body_user.strip():
            user_id = body_user.strip()

    user_name: str | None = None
    raw_name = headers.get(OWUI_USER_NAME_HEADER)
    if raw_name:
        try:
            decoded = urllib.parse.unquote(raw_name).strip()
        except Exception:
            decoded = raw_name.strip()
        if decoded:
            user_name = decoded

    return user_id, user_name


# ---------- auxiliary-call detection ----------
#
# OWUI sends several "internal" /v1/chat/completions calls that aren't real
# user-facing chat — title generation, tag generation, follow-up suggestions,
# autocomplete. These come with small max_tokens and don't need the Sohn
# persona, the cortex memories, or any tools. Forcing them through the full
# pipeline causes the model to over-deliberate (content=null,
# finish_reason=length) and fans out cortex memory-extraction tasks for no
# reason.
#
# We detect them by max_tokens. Real Sohn chats either don't set max_tokens
# or set a generous budget (≥1024). Auxiliary calls cap at 50-500.
AUXILIARY_MAX_TOKENS_THRESHOLD = int(
    os.environ.get("ACTI_AUXILIARY_MAX_TOKENS_THRESHOLD", "800")
)


def _is_auxiliary_call(body: dict) -> bool:
    """True if the request looks like an OWUI auxiliary call (title gen, tag
    gen, follow-up suggestion, etc.) that should bypass Sohn injection,
    tools, and cortex extraction."""
    mt = body.get("max_tokens")
    if isinstance(mt, int) and 0 < mt < AUXILIARY_MAX_TOKENS_THRESHOLD:
        return True
    return False


# ---------- app lifespan ----------

def _purge_legacy_global_memory(memory_dir: Path) -> int:
    """Delete *.md files at the root of memory_dir.

    These are pre-partitioning global memories that were visible to every
    user (the cross-account leak). Per-user partitioning makes them
    functionally orphaned — we can't attribute any of them to a single
    user, and the new inject path can no longer reach them anyway, so
    deleting is the correct action. Files inside `users/` subdirectories
    are not touched. Idempotent: a clean dir is a no-op.
    """
    if not memory_dir.exists():
        return 0
    legacy = [p for p in memory_dir.iterdir() if p.is_file() and p.suffix == ".md"]
    deleted = 0
    for p in legacy:
        try:
            p.unlink()
            deleted += 1
        except OSError as e:
            _log(f"[gateway] failed to delete legacy memory {p.name}: {e}")
    return deleted


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
    deleted = _purge_legacy_global_memory(MEMORY_DIR)
    if deleted:
        _log(
            f"[gateway] WARNING: deleted {deleted} legacy global memory file(s). "
            f"These were pre-partitioning artifacts visible to ALL users. "
            f"Per-user partitioning is now active."
        )
    user_root = MEMORY_DIR / "users"
    n_users = len([p for p in user_root.glob("*") if p.is_dir()]) if user_root.exists() else 0
    _log(
        f"[gateway] started. auth={'ON' if app.state.api_keys else 'OFF (dev)'} "
        f"keys_loaded={len(app.state.api_keys)} "
        f"skills_loaded={list(app.state.skills.keys()) or 'none'} "
        f"media={'ON ('+media.LUMEN_BASE_URL+')' if media.media_enabled() else 'OFF'} "
        f"library={'ON ('+library.LIBRARY_BASE_URL+')' if library.library_enabled() else 'OFF'} "
        f"memory={MEMORY_DIR} ({n_users} user partitions, "
        f"extraction={'ON' if MEMORY_EXTRACTION_ENABLED else 'OFF'})"
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


# Style guard appended to OWUI's auxiliary-call system messages. Without it,
# OWUI's default title prompt explicitly asks the model to "Generate a … title
# with an emoji", which violates Sohn's no-emoji branding. The full Sohn prompt
# isn't injected for auxiliary calls (too expensive for sub-800-token budgets),
# so this one-liner is the cheapest way to keep titles / tags / follow-ups in
# the same plain-text register as the chat itself.
_AUXILIARY_STYLE_GUARD = (
    "Output plain text only. No emojis, no decorative characters, no quotation "
    "marks around the result. Match the user's language."
)

# Strip any "with an emoji" / "include emoji" clause from the existing system
# prompt before appending the guard. The model otherwise has to reconcile two
# contradictory instructions and sometimes resolves toward the more specific
# (emoji-positive) one. Removing the conflict at the source is more reliable
# than trusting last-instruction-wins.
_EMOJI_INSTRUCTION_RE = re.compile(
    r"(?i)\s*(?:with|including|using|plus|and|featuring)\s*(?:an?\s+)?emojis?\b"
)


def _augment_auxiliary_messages(messages: list) -> list:
    """Strip any pro-emoji clause from the system message and append the
    style guard. Falls back to prepending a new system message when the
    request didn't include one."""
    if messages and messages[0].get("role") == "system":
        sys_msg = messages[0]
        existing = sys_msg.get("content", "") or ""
        if isinstance(existing, list):
            existing = "\n".join(
                p.get("text", "") for p in existing if isinstance(p, dict)
            )
        cleaned = _EMOJI_INSTRUCTION_RE.sub("", str(existing))
        merged = cleaned.rstrip() + "\n\n" + _AUXILIARY_STYLE_GUARD
        return [{**sys_msg, "content": merged}] + messages[1:]
    return [{"role": "system", "content": _AUXILIARY_STYLE_GUARD}] + messages


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

    # Auxiliary call (title gen, tag gen, follow-up suggestions) — bypass the
    # Sohn pipeline entirely. These don't need persona, memories, tools, or
    # extraction. Letting them through the full pipeline causes the model to
    # over-deliberate against a small token budget and OWUI sees empty
    # responses (the "infinite loop" UX).
    if _is_auxiliary_call(body):
        # Auxiliary calls have small token budgets and shouldn't think — they
        # need short, mechanical responses (titles, tags, follow-up suggestions).
        # With thinking on, the model spends the entire budget deliberating and
        # never reaches content. Force thinking off here only — this targeted
        # disable does NOT affect the user-facing chat path.
        ctk = body.get("chat_template_kwargs") or {}
        if "enable_thinking" not in ctk:
            ctk["enable_thinking"] = False
            body["chat_template_kwargs"] = ctk
        # Append the no-emoji style guard. OWUI's default title prompt
        # asks for emojis explicitly; without this guard they leak into
        # chat titles, tags, and follow-up suggestions in the sidebar.
        body["messages"] = _augment_auxiliary_messages(body.get("messages", []))
        # Auxiliary requests must not carry the OWUI user identifier —
        # the engine doesn't need it and it's not part of the chat itself.
        body.pop("user", None)
        _log(f"[gateway] auxiliary call (max_tokens={body.get('max_tokens')}) — pass-through, thinking off, style-guarded")
        client: httpx.AsyncClient = request.app.state.client
        if stream:
            async def aux_stream():
                try:
                    async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                except httpx.HTTPError as e:
                    msg = str(e).replace('"', "'")
                    yield (f'data: {{"error":{{"message":"{msg}","type":"upstream_error"}}}}\n\n').encode()
            return StreamingResponse(aux_stream(), media_type="text/event-stream")
        try:
            resp = await client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as e:
            return _error(502, f"Upstream error: {e}", "upstream_error")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    skills.maybe_reload_skills(request.app.state)
    skill_lib: dict = request.app.state.skills
    client_passed_tools = bool(body.get("tools"))

    # The chat UI sends a `features` block to indicate which native toggles
    # are on (image_generation, code_interpreter, web_search, video_generation).
    # When image OR video generation is toggled, the chat UI handles the gen
    # call directly and posts the result back into the message — we must NOT
    # also let the model invoke our generate_* tool or we'd get duplicate output.
    features = body.get("features") or {}
    suppress_image = bool(features.get("image_generation"))
    suppress_video = bool(features.get("video_generation"))

    # Reserved tool names — always owned by the proxy regardless of who tried
    # to register them in body["tools"]. Prevents a user-defined OWUI tool
    # named (say) `generate_image` from shadowing our authoritative version.
    OWNED_TOOLS = {"load_skill", "generate_image", "generate_video", "recall_context"}
    if client_passed_tools:
        client_tool_names = {
            (t.get("function") or {}).get("name") for t in body.get("tools") or []
        }
        collisions = client_tool_names & OWNED_TOOLS
        if collisions:
            _log(f"[gateway] reserved tool name collision: {collisions} — proxy keeps authority")
            # Drop the client's stale schemas; we'll re-add ours below.
            body["tools"] = [
                t for t in body.get("tools") or []
                if (t.get("function") or {}).get("name") not in OWNED_TOOLS
            ]
            client_passed_tools = bool(body["tools"])  # may now be empty

    # Skill / media / library tools activate when the caller didn't bring its own.
    use_skills = bool(skill_lib) and not client_passed_tools
    use_media = media.media_enabled() and not client_passed_tools
    use_library = library.library_enabled() and not client_passed_tools
    use_proxy_tools = use_skills or use_media or use_library

    body["messages"] = _inject_system_prompt(
        body.get("messages", []),
        skills_manifest=request.app.state.skills_manifest if use_skills else "",
    )

    # Cortex: per-user memory partitioning. Identity resolution prefers the
    # OWUI-forwarded headers (X-OpenWebUI-User-Id / X-OpenWebUI-User-Name),
    # which OWUI sets when ENABLE_FORWARD_USER_INFO_HEADERS=true is exported
    # in launch_owui.sh. Without those, falls back to body["user"] (OpenAI-
    # standard) and finally to a shared _anonymous partition.
    raw_user_id, raw_user_name = _resolve_user_identity(request, body)
    user_mem_dir = cortex.user_memory_dir(MEMORY_DIR, raw_user_id)
    is_first_interaction = not cortex.has_any_memory(user_mem_dir)

    # When the OWUI signup form supplied a full name and this is a fresh user,
    # seed it directly as a user memory. The model will see "User's full name
    # is X" via inject_memories on this very turn — no onboarding back-and-forth
    # needed. If no name was forwarded, fall back to the model-driven onboarding
    # block so it asks the user explicitly.
    name_seeded = False
    if is_first_interaction and raw_user_name:
        name_seeded = cortex.seed_signup_name(user_mem_dir, raw_user_name)
    include_onboarding = is_first_interaction and not name_seeded

    body["messages"] = cortex.inject_memories(
        body["messages"], user_mem_dir, include_onboarding=include_onboarding,
    )
    # Don't forward the OWUI user id to the engine — it's a routing/storage
    # identifier, not part of the chat itself.
    body.pop("user", None)
    if raw_user_id:
        _log(
            f"[gateway] user partition: {cortex.sanitize_user_id(raw_user_id)} "
            f"(name={'seeded' if name_seeded else ('forwarded' if raw_user_name else 'unknown')}, "
            f"onboarding={'YES' if include_onboarding else 'no'})"
        )
    else:
        _log("[gateway] no user identifier in headers/body — using shared _anonymous partition")

    client: httpx.AsyncClient = request.app.state.client

    # Cortex hooks for the agent loop.
    async def _between_turns(msgs: list[dict]) -> list[dict]:
        # Exact token count from the engine's own tokenizer (~1-2ms).
        # Accounts for chat-template overhead the char estimate misses.
        triggered, n_tok = await cortex.should_compact_exact(
            client, msgs,
            max_tokens=CONTEXT_WINDOW_TOKENS,
            buffer_tokens=COMPACT_BUFFER_TOKENS,
            served_name=SERVED_NAME,
        )
        if triggered:
            _log(f"[gateway] compaction triggered: {n_tok:,} >= {CONTEXT_WINDOW_TOKENS-COMPACT_BUFFER_TOKENS:,}")
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
            client=client, messages=list(msgs), memory_dir=user_mem_dir,
            served_name=SERVED_NAME,
        ))

    if use_proxy_tools:
        # Build the registry of tools that load_tool is allowed to activate.
        # OWUI feature toggles can suppress generate_image / generate_video
        # entirely (they get handled by OWUI native paths instead).
        allowed_schemas = _real_tool_schemas()
        if not use_skills:
            allowed_schemas.pop("load_skill", None)
        if not use_media or suppress_image:
            allowed_schemas.pop("generate_image", None)
        if not use_media or suppress_video:
            allowed_schemas.pop("generate_video", None)
        if not use_library:
            allowed_schemas.pop("recall_context", None)

        if suppress_image or suppress_video:
            _log(f"[gateway] feature toggles active: "
                 f"image_gen={'OWUI' if suppress_image else 'agent'} "
                 f"video_gen={'OWUI' if suppress_video else 'agent'}")

        # Only the bootstrap tool is exposed by default. Real schemas land
        # in body["tools"] only after the model calls load_tool with the
        # tool's name.
        body["tools"] = [LOAD_TOOL_SCHEMA]
        body.setdefault("tool_choice", "auto")

        handlers = _build_tool_handlers(skill_lib)
        handlers["load_tool"] = _make_load_tool_handler(body, allowed_schemas)
        # Drop matching handlers for suppressed tools too, so a stale
        # tool_call from an earlier turn can't accidentally re-invoke a
        # suppressed tool — and the model can't load_tool its way around
        # the suppression either.
        for tname in list(handlers):
            if tname not in allowed_schemas and tname != "load_tool":
                handlers.pop(tname, None)
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

@app.post("/v1/videos/generations")
async def videos_generations(request: Request):
    """OpenAI-style video generation (custom endpoint, not in OpenAI's API).

    Request body shape (intentionally permissive — accepts both our own
    fields and the OpenAI image-style fields where they map):

        {
          "prompt":        str,               # required
          "duration":      int,               # 1-8, default 4
          "size":          "1920x1080" | ... | optional
          "resolution":    "540p"|"720p"|"1080p", default "1080p"
          "aspect_ratio":  "16:9"|"9:16",     default "16:9"
          "camera_motion": "none"|"dolly_in"|..., default "none"
        }

    Response:

        { "created": int, "data": [{"url": "/media/<sha>.mp4"}] }
    """
    _require_auth(request)
    if not media.media_enabled():
        return _error(503, "video generation is not configured on this deployment.",
                      "service_unavailable")
    try:
        body = await request.json()
    except Exception as e:
        return _error(400, f"Invalid JSON body: {e}", "invalid_request_error")

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return _error(400, "`prompt` is required", "invalid_request_error")

    duration = max(1, min(int(body.get("duration", 4)), 8))
    resolution = body.get("resolution") or "1080p"
    aspect_ratio = body.get("aspect_ratio") or "16:9"
    camera_motion = body.get("camera_motion") or "none"
    # Accept OpenAI-style "size" → infer resolution + aspect_ratio when given
    size = body.get("size")
    if isinstance(size, str) and "x" in size:
        try:
            w, h = (int(x) for x in size.split("x", 1))
            aspect_ratio = "9:16" if h > w else "16:9"
            short_side = min(w, h)
            resolution = "540p" if short_side <= 540 else (
                "720p" if short_side <= 720 else "1080p"
            )
        except ValueError:
            pass

    result = await media.generate_video_raw(
        prompt=prompt, duration=duration, resolution=resolution,
        aspect_ratio=aspect_ratio, camera_motion=camera_motion,
    )
    if "error" in result:
        return _error(502, result["error"], "upstream_error")
    return {
        "created": int(time.time()),
        "data": [{"url": result["url"], "revised_prompt": prompt}],
    }


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
    """Liveness + (optional) detail probe.

    Default returns the minimal envelope for orchestrators / nginx checks.
    `?detail=1` adds the breakdown the status page renders — engine state,
    skill catalogue size, active per-user memory partitions, and which
    optional integrations are wired up.
    """
    client: httpx.AsyncClient = request.app.state.client
    engine_ok = False
    try:
        r = await client.get("/health", timeout=5.0)
        engine_ok = r.status_code == 200
    except Exception:
        engine_ok = False

    base = {
        "status": "ok" if engine_ok else ("degraded" if engine_ok is False else "down"),
        "model": SERVED_NAME,
        "version": "0.1.0",
    }
    if request.query_params.get("detail") not in ("1", "true", "yes"):
        return base

    # Detail block — cheap reads, all in-process.
    user_root = MEMORY_DIR / "users"
    try:
        partition_count = sum(1 for p in user_root.iterdir() if p.is_dir()) if user_root.exists() else 0
    except OSError:
        partition_count = 0

    skills = getattr(request.app.state, "skills", {}) or {}
    base["detail"] = {
        "engine": "ok" if engine_ok else "down",
        "proxy": "ok",  # we got here, so proxy is up
        "skills_loaded": len(skills),
        "skill_names": sorted(skills.keys()),
        "memory_partitions": partition_count,
        "memory_extraction": MEMORY_EXTRACTION_ENABLED,
        "media_enabled": media.media_enabled(),
        "library_enabled": library.library_enabled(),
        "auth_enforced": bool(request.app.state.api_keys),
    }
    return base


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, host="0.0.0.0",
        port=int(os.environ.get("SOHN_PROXY_PORT", 8080)),
        log_level="info",
    )
