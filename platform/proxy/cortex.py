"""Cortex — ACTi's memory layer.

Two responsibilities, both ported from Anthropic Claude Code's
src/services/compact/ and src/services/extractMemories/:

  1. **In-session compaction**. When a conversation grows past the
     model's effective context window, summarize older turns into a
     structured 9-section block and reconstruct the message list as
     [system, summary as user msg, recent K turns]. Reuses the leaked
     Claude Code prompt verbatim — Anthropic tuned it on millions of
     agent runs and we'd be foolish to write our own.

  2. **Cross-session auto-memory**. After each agent turn, a
     fire-and-forget background task analyses the last few messages
     and writes/updates `*.md` files in `MEMORY_DIR`, plus a one-line
     index entry in `MEMORY.md`. The index is injected into every
     subsequent system prompt. Memory types: user / feedback / project
     / reference.

PUBLIC API

    def should_compact(messages, *, max_tokens) -> bool
        Fast (microseconds) char-based token estimate. Call between
        agent turns; returns True if we're near the engine's context cap.

    async def compact(*, client, messages, served_name, ...) -> list[dict]
        Slow (~5–10s) — calls the engine to summarize older turns and
        returns a reconstructed message list. Only call when
        should_compact() is True.

    def inject_memories(messages, memory_dir) -> list[dict]
        Synchronously load MEMORY.md and prepend it to the system
        prompt. Cheap — read-once-and-cache.

    async def extract_memories(*, client, messages, memory_dir, ...) -> int
        Background task. LLM call extracts 0–N facts from the last few
        turns and writes them to MEMORY_DIR. Best-effort; returns the
        number of memory files written.

DESIGN NOTES

  - Token counting uses the engine's own tokenizer via /v1/messages/
    count_tokens (1–2ms localhost call, exact down to the chat template).
    A char-based heuristic (`chars / 3.5`) is kept as a synchronous
    fallback for tests and offline tools.
  - Memory files have YAML-ish frontmatter (name, description, type).
  - All memory writes go through atomic-rename for crash safety.
  - The compaction call REUSES the engine's prefix cache by sending
    the same system prompt + same messages, with the compaction
    instruction appended as a final user message. SGLang's
    RadixAttention hits the cache → cheap.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from cortex_prompts import (
    BASE_COMPACT_PROMPT,
    CONTINUATION_PREAMBLE,
    NO_TOOLS_PREAMBLE,
    memory_extract_prompt,
)


# ---------- configuration ----------

# Approximate tokens per character. English averages ~3.5 chars/token for
# the BPE tokenizers we serve. Off by maybe 10% in either direction;
# that's fine since the compact threshold has a generous buffer.
CHARS_PER_TOKEN = 3.5

# Default safety buffer between estimated tokens and engine context cap.
# Must be > one full assistant reply + tool call. Claude Code uses 13_000;
# we use 16_000 for headroom on the engine's long-context window.
DEFAULT_BUFFER_TOKENS = 16_000

# Number of recent turns kept verbatim post-compaction. A "turn" here is
# one user message + the assistant response (and any tool messages
# between). Lower = more context freed; higher = more recent fidelity.
DEFAULT_KEEP_RECENT = 6


def _default_log(msg: str) -> None:
    print(msg, file=sys.stdout, flush=True)


# ---------- token counting ----------

def estimate_tokens(messages: list[dict]) -> int:
    """Char-based fallback estimate. Microseconds-fast, ±10% accuracy.

    Used when the engine isn't available (offline tests) or as a quick
    sanity bound. For real production checks, prefer count_tokens()
    which calls the engine's tokenizer.
    """
    total_chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    total_chars += len(part.get("text", ""))
        tc = m.get("tool_calls")
        if tc:
            try:
                total_chars += len(json.dumps(tc))
            except Exception:
                pass
    return int(total_chars / CHARS_PER_TOKEN)


async def count_tokens(
    client: httpx.AsyncClient,
    messages: list[dict],
    served_name: str = "Sohn",
) -> int:
    """Exact token count from the engine's tokenizer.

    Calls SGLang's /v1/messages/count_tokens — accounts for chat template
    overhead (role tokens, tool schemas, system prompt boilerplate) that
    the char-based estimate misses entirely. Localhost call, ~1–2 ms.

    Falls back to estimate_tokens() if the request fails.
    """
    try:
        resp = await client.post(
            "/v1/messages/count_tokens",
            json={"model": served_name, "messages": messages},
            timeout=5.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            n = data.get("input_tokens")
            if isinstance(n, int):
                return n
    except (httpx.HTTPError, ValueError):
        pass
    return estimate_tokens(messages)


# ---------- compaction ----------

def should_compact(
    messages: list[dict],
    *,
    max_tokens: int = 256_000,
    buffer_tokens: int = DEFAULT_BUFFER_TOKENS,
) -> bool:
    """Synchronous predicate using the char-based estimate.

    Microseconds-fast, ±10% accuracy. Useful as a cheap pre-check before
    paying the (still small) HTTP round-trip cost of the exact counter.
    For production, prefer should_compact_exact() — the char estimator
    can undercount by ~2x once chat template overhead is included.
    """
    return estimate_tokens(messages) >= (max_tokens - buffer_tokens)


async def should_compact_exact(
    client: httpx.AsyncClient,
    messages: list[dict],
    *,
    max_tokens: int = 256_000,
    buffer_tokens: int = DEFAULT_BUFFER_TOKENS,
    served_name: str = "Sohn",
) -> tuple[bool, int]:
    """Exact predicate using the engine's own tokenizer.

    Returns (should_compact, current_token_count). The token count is
    returned alongside the bool so the caller can log it without a
    second tokenize call.
    """
    n = await count_tokens(client, messages, served_name=served_name)
    return n >= (max_tokens - buffer_tokens), n


async def compact(
    *,
    client: httpx.AsyncClient,
    messages: list[dict],
    served_name: str = "Sohn",
    keep_recent: int = DEFAULT_KEEP_RECENT,
    log: Callable[[str], None] = _default_log,
) -> list[dict]:
    """Replace older turns with a structured 9-section summary.

    Strategy:
      1. Split `messages` into [system, ..._old, ..._recent] where
         `_recent` is the last `keep_recent` non-system messages.
      2. Call the engine with `_old + [user: NO_TOOLS_PREAMBLE +
         BASE_COMPACT_PROMPT]` and `tools=[]`. The engine returns a
         9-section summary.
      3. Reconstruct: `[system, summary as user msg with continuation
         preamble, ..._recent]`.

    Reuses the same system prompt + messages prefix → SGLang's
    RadixAttention hits the cache for the long prefix.

    Returns the reconstructed messages. On error, returns the original
    list unchanged (compaction is best-effort — better to retain
    full context than break the request).
    """
    if not messages:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    body_msgs = [m for m in messages if m.get("role") != "system"]
    if len(body_msgs) <= keep_recent + 1:
        # Not enough older turns to compact — nothing useful to do.
        return messages

    older = body_msgs[:-keep_recent]
    recent = body_msgs[-keep_recent:]

    summarization_messages = (
        system_msgs
        + older
        + [{"role": "user", "content": NO_TOOLS_PREAMBLE + "\n\n" + BASE_COMPACT_PROMPT}]
    )
    request = {
        "model": served_name,
        "messages": summarization_messages,
        "stream": False,
        "max_tokens": 4_000,
        "temperature": 0.3,  # lower temp → more faithful summary
        # Force thinking off for the summary itself — we don't want a
        # long reasoning preamble before the structured output.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.time()
    in_tokens = estimate_tokens(messages)
    try:
        resp = await client.post(
            "/v1/chat/completions", json=request, timeout=120.0,
        )
    except httpx.HTTPError as e:
        log(f"[cortex] compaction call failed: {e}; returning original messages")
        return messages
    if resp.status_code != 200:
        log(f"[cortex] compaction status={resp.status_code}; returning original messages")
        return messages

    data = resp.json()
    summary = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not summary.strip():
        log("[cortex] compaction returned empty summary; returning original messages")
        return messages

    # Reconstruct: system(s) + summary-as-user + recent.
    summary_msg = {
        "role": "user",
        "content": CONTINUATION_PREAMBLE + summary.strip() + "\n",
    }
    new_messages = system_msgs + [summary_msg] + recent

    out_tokens = estimate_tokens(new_messages)
    log(
        f"[cortex] compacted: {len(messages)} msgs ({in_tokens:,} tok) → "
        f"{len(new_messages)} msgs ({out_tokens:,} tok) in {time.time()-t0:.1f}s"
    )
    return new_messages


# ---------- memory directory ----------

# Memory file frontmatter — minimal YAML-ish parser (same as skills.py).

def _parse_frontmatter(text: str) -> tuple[dict, str]:
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


def _index_path(memory_dir: Path) -> Path:
    return memory_dir / "MEMORY.md"


def read_index(memory_dir: Path, *, max_lines: int = 200) -> str:
    """Return MEMORY.md contents (truncated to max_lines), or '' if absent."""
    p = _index_path(memory_dir)
    if not p.exists():
        return ""
    try:
        text = p.read_text()
    except OSError:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [
            f"<!-- index truncated at {max_lines} lines; older entries dropped -->"
        ]
    return "\n".join(lines)


# ---------- per-user partitioning ----------
#
# Memory is partitioned per end-user. Each user's memories live under a
# dedicated subdirectory derived from the OpenAI-standard `user` field of
# the chat completion request. Without partitioning, one user's "my name
# is Neel" memory file would be injected into another user's system prompt,
# causing cross-account identity leakage.

# Allowed chars in a directly-usable user_id segment. Anything outside this
# set forces a hash fallback so weird inputs (emails with @, plus signs,
# unicode, path traversal attempts) cannot escape the per-user namespace.
_USER_ID_RE = re.compile(r"[^A-Za-z0-9._-]")
_USER_ID_MAX_LEN = 64


def sanitize_user_id(user_id: str | None) -> str:
    """Map a raw user identifier to a stable, filesystem-safe segment.

    Same input always produces the same output. Inputs containing chars
    outside [A-Za-z0-9._-] or longer than 64 chars are mapped through a
    sha256 prefix so the result stays bounded and traversal-safe.
    Returns "_anonymous" for empty / None inputs.
    """
    if user_id is None:
        return "_anonymous"
    if not isinstance(user_id, str):
        user_id = str(user_id)
    user_id = user_id.strip()
    if not user_id:
        return "_anonymous"
    cleaned = _USER_ID_RE.sub("_", user_id)
    if cleaned != user_id or len(cleaned) > _USER_ID_MAX_LEN:
        h = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        # Build a strict alphanumeric prefix for human readability — keep only
        # [A-Za-z0-9] so no '.' / '..' / '_' run can survive into the segment.
        prefix = re.sub(r"[^A-Za-z0-9]", "", cleaned)[:16] or "u"
        return f"{prefix}_{h}"
    # Even when the input is "clean", reject standalone ".." or "." (which match
    # the [A-Za-z0-9._-] charclass but are filesystem-special).
    if cleaned in ("", ".", ".."):
        h = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        return f"u_{h}"
    return cleaned


def user_memory_dir(base: Path, user_id: str | None) -> Path:
    """Per-user memory directory under `<base>/users/<sanitized_id>/`.

    Always returns a path; callers are responsible for mkdir before write.
    Reads on a non-existent dir are handled gracefully by read_index().
    """
    return base / "users" / sanitize_user_id(user_id)


def has_any_memory(memory_dir: Path) -> bool:
    """True if this user has at least one memory file (excluding the index)."""
    if not memory_dir.exists():
        return False
    for p in memory_dir.glob("*.md"):
        if p.name == "MEMORY.md":
            continue
        return True
    return False


# ---------- memory injection ----------

_ONBOARDING_BLOCK = (
    "\n\n## First interaction with this user\n\n"
    "You have not spoken with this user before — you do not know their name "
    "or anything about them yet. Your very first response in this conversation "
    "must:\n"
    "  1. Briefly introduce yourself in one short line.\n"
    "  2. Politely ask the user for their full name so you can address them "
    "properly going forward.\n"
    "Once they share it, acknowledge it briefly and continue with their actual "
    "request. If they decline, sidestep, or ask you to skip the introduction, "
    "do not press — proceed naturally and address them however they prefer. "
    "Do not invent a name, do not assume a name from any other source, and do "
    "not address the user by a name they have not given you in THIS conversation.\n"
)


def inject_memories(
    messages: list[dict],
    memory_dir: Path,
    *,
    include_onboarding: bool = False,
) -> list[dict]:
    """Prepend MEMORY.md and/or an onboarding block to the system prompt.

    Idempotent: if neither memories nor onboarding apply, returns
    `messages` unchanged. Safe to call on every request.

    `include_onboarding=True` injects a one-time first-interaction block
    that tells the model to ask for the user's full name. Caller decides
    when to set this — typically when `has_any_memory(memory_dir)` is False.
    """
    index = read_index(memory_dir)
    parts: list[str] = []
    if index.strip():
        parts.append(
            "\n\n## Memory\n\n"
            "The following memories from prior sessions may be relevant. Each "
            "is a one-line pointer to a memory file. When directly relevant, "
            "internalise them silently — do not narrate the recall.\n\n"
            + index
            + "\n"
        )
    if include_onboarding:
        parts.append(_ONBOARDING_BLOCK)
    if not parts:
        return messages

    block = "".join(parts)

    if messages and messages[0].get("role") == "system":
        sys_msg = messages[0]
        existing = sys_msg.get("content", "") or ""
        if isinstance(existing, list):
            existing = "\n".join(
                p.get("text", "") for p in existing if isinstance(p, dict)
            )
        new_sys = {**sys_msg, "content": str(existing).rstrip() + block}
        return [new_sys] + messages[1:]
    return [{"role": "system", "content": block.lstrip()}] + messages


# ---------- memory extraction ----------

# Sanitization: lowercase + alphanum + underscores.
_NAME_RE = re.compile(r"[^a-z0-9_]+")
_FILE_RE = re.compile(r"^---\n(.+?)\n---\n(.+)$", re.DOTALL)


def _sanitize_name(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    name = _NAME_RE.sub("", name)
    return name[:64] or "memory"


_FENCE_RE = re.compile(r"```(?:markdown|md)?\n?(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if the model wrapped the file in them."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _atomic_write(path: Path, content: str) -> None:
    """Write a file atomically via tmp+rename, so partial writes can't
    leave a memory file half-written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex[:8]}")
    tmp.write_text(content)
    tmp.replace(path)


async def extract_memories(
    *,
    client: httpx.AsyncClient,
    messages: list[dict],
    memory_dir: Path,
    served_name: str = "Sohn",
    new_message_count: int = 6,
    log: Callable[[str], None] = _default_log,
) -> int:
    """Best-effort memory extraction from the most recent messages.

    Calls the engine with the conversation + an extraction instruction.
    Parses the response — looking for one or more FILE blocks of the form:

        FILE: <name>.md
        ---
        name: ...
        description: ...
        type: ...
        ---
        <body>
        END FILE

    Writes each block to memory_dir/<name>.md and updates MEMORY.md.

    Returns the number of memory files written. Errors are swallowed
    (this runs in the background and must never break the user's turn).
    """
    if not messages:
        return 0

    instruction = memory_extract_prompt(new_message_count)
    request = {
        "model": served_name,
        "messages": list(messages) + [{
            "role": "user",
            "content": (
                instruction
                + "\n\n## Output format\n\n"
                "Output ZERO or more memory files in this exact format:\n\n"
                "```\n"
                "FILE: <filename>.md\n"
                "---\n"
                "name: <label>\n"
                "description: <one-line>\n"
                "type: user | feedback | project | reference\n"
                "---\n"
                "<body>\n"
                "END FILE\n"
                "```\n\n"
                "Then output ONE final block:\n\n"
                "```\n"
                "INDEX:\n"
                "- [<title>](<filename>.md) — <one-line hook>\n"
                "...\n"
                "END INDEX\n"
                "```\n\n"
                "If nothing is worth saving, output exactly:\n\n"
                "NO_MEMORIES\n"
            ),
        }],
        "stream": False,
        "max_tokens": 3_000,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.time()
    try:
        resp = await client.post("/v1/chat/completions", json=request, timeout=60.0)
    except httpx.HTTPError as e:
        log(f"[cortex] memory extraction call failed: {e}")
        return 0
    if resp.status_code != 200:
        log(f"[cortex] memory extraction status={resp.status_code}")
        return 0

    data = resp.json()
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if "NO_MEMORIES" in text and "FILE:" not in text:
        log(f"[cortex] no memories to save (in {time.time()-t0:.1f}s)")
        return 0

    written = 0
    file_blocks = re.findall(
        r"FILE:\s*([^\n]+)\n(.*?)END FILE",
        text, re.DOTALL,
    )
    for filename, content in file_blocks:
        filename = filename.strip()
        if not filename.endswith(".md") or "/" in filename or ".." in filename:
            continue
        body = content.strip()
        # Sanity-check the frontmatter is present
        if not body.startswith("---"):
            continue
        meta, _ = _parse_frontmatter(body)
        if not meta.get("type") in {"user", "feedback", "project", "reference"}:
            continue
        try:
            _atomic_write(memory_dir / filename, body + "\n")
            written += 1
        except OSError as e:
            log(f"[cortex] failed to write {filename}: {e}")

    # Update index. Capture an INDEX block if present; otherwise append
    # one-line entries derived from the frontmatter of each file.
    index_match = re.search(r"INDEX:\s*\n(.*?)END INDEX", text, re.DOTALL)
    if index_match and written > 0:
        index_lines = [l for l in index_match.group(1).splitlines() if l.strip()]
        existing = read_index(memory_dir)
        # Merge: drop lines pointing to filenames that just got rewritten,
        # then append the new lines.
        rewritten_names = {fn.strip() for fn, _ in file_blocks}
        merged_lines = []
        for line in existing.splitlines():
            if any(f"({n})" in line for n in rewritten_names):
                continue
            merged_lines.append(line)
        merged_lines.extend(index_lines)
        try:
            _atomic_write(_index_path(memory_dir), "\n".join(merged_lines) + "\n")
        except OSError as e:
            log(f"[cortex] failed to update MEMORY.md: {e}")

    log(f"[cortex] memory extraction: wrote {written} files in {time.time()-t0:.1f}s")
    return written
