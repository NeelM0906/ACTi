"""Spark — ACTi's streaming agent loop.

A small, opinionated agent runtime that brokers between an OpenAI-compatible
inference engine and a streaming client (OWUI, OpenAI SDK, etc). Lives between
the two:

    client  ←──SSE──  Spark  ──HTTP──→  inference engine
                       │
                       └──tool calls──→ caller-supplied handlers

The engine streams tokens (content, reasoning_content, tool_call deltas);
Spark forwards content/reasoning verbatim, intercepts tool_call deltas,
dispatches them to caller-supplied handlers, threads the results back into
the conversation, and loops until the model produces a turn without any
tool call.

ARCHITECTURE — producer/consumer

    ┌───────────────────────────────────────────────────────────┐
    │ run_agent_stream() — async generator returned to caller   │
    │                                                           │
    │   ┌─ background runner task ────────┐    ┌──────────────┐ │
    │   │ for turn in range(max_turns):   │    │ asyncio      │ │
    │   │   stream engine → push chunks ──┼──→ │  Queue       │─┼─→ yield to client
    │   │   if tool_calls:                │    │              │ │
    │   │     emit status chunk           │    │              │ │
    │   │     start progress task ────────┼──→ │  (every 15s) │ │
    │   │     await handler(args) ────────┼──→ │              │ │
    │   │     append result, loop         │    │              │ │
    │   └─────────────────────────────────┘    └──────────────┘ │
    └───────────────────────────────────────────────────────────┘

The producer/consumer split decouples engine reads from client writes so
neither blocks the other, and lets us inject synthetic chunks (status,
progress, keepalive, errors) freely without disturbing the engine stream.

KEY DESIGN CHOICES

  - Tools registered on body["tools"] stay registered across every turn.
    Never strip them mid-conversation — that's what caused the historical
    chat-template XML leak (tool_call messages render as raw XML when the
    schema isn't in the template's tools list).

  - Tool_call deltas are accumulated by `index`, NEVER by `id`. Only the
    first delta per call carries `id`/`function.name`; subsequent deltas
    carry only `index` plus partial `function.arguments`. This is the
    LiteLLM #20711 / OpenAI streaming-spec gotcha that causes silent
    argument loss in naive implementations.

  - Periodic progress chunks during long tool execution serve as both UX
    feedback AND SSE keepalive. No separate `: ping` comment lines.

  - max_turns hard-caps the loop (default 4). Per-turn structured logs
    isolate failures to a specific turn + tool.

PUBLIC API

    async def run_agent_stream(
        *,
        client: httpx.AsyncClient,
        body: dict,
        tool_handlers: dict[str, ToolHandler],
        tool_labels: dict[str, ToolLabels] | None = None,
        served_name: str = "agent",
        max_turns: int = 4,
        keepalive_interval_s: float = 15.0,
        log: Callable[[str], None] = print,
    ) -> AsyncGenerator[bytes, None]: ...

The caller is responsible for:
  - registering tool schemas in body["tools"] before invoking
  - providing async handler(args) -> str for each advertised tool
  - choosing the engine endpoint via the httpx client base_url

Spark handles:
  - SSE chunk parsing (OpenAI-compatible wire format)
  - tool_call delta accumulation by index
  - tool dispatch + result threading back into messages
  - status / progress chunks during tool execution
  - graceful error envelopes
  - max_turns enforcement
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from typing import Awaitable, Callable, NamedTuple

import httpx


def _default_log(line: str) -> None:
    """Write a log line to stdout, force-flushed.

    `print()`'s default buffering swallows lines when stdout is piped
    (e.g. tee'd to a log file by the launcher), so we always flush.
    """
    print(line, file=sys.stdout, flush=True)


# ---------- content scrubber ----------

class _ToolCallStripper:
    """Strip <tool_call>…</tool_call> blocks from streamed content.

    The engine's tool-call parser occasionally lets a malformed tool-call
    template through as plain content text — typically when the model emits
    a tool call under sustained adversarial pressure (e.g. trying to call
    recall_context to answer "what model are you") and the chat template's
    parser doesn't catch the variant. The leaked block looks like:

        <tool_call>function=recall_context>
        <parameter=query>…</parameter>
        </function>
        </tool_call>

    Without filtering, the user sees raw XML in the chat stream. We can't
    rescue these into structured tool calls reliably (the format varies),
    so we just drop them.

    State machine: PASS -> (sees OPEN) -> SUPPRESS -> (sees CLOSE) -> PASS.
    Held lookahead in PASS state prevents emitting partial tag prefixes.
    """

    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"
    _LOOKAHEAD = max(len(OPEN), len(CLOSE))

    def __init__(self) -> None:
        self._buf = ""
        self._suppressing = False
        self.bytes_dropped = 0

    def feed(self, chunk: str) -> str:
        """Process one content chunk. Returns the cleaned text to forward."""
        out: list[str] = []
        self._buf += chunk
        while True:
            if self._suppressing:
                idx = self._buf.find(self.CLOSE)
                if idx == -1:
                    # Drop everything except a trailing slice that might be
                    # the start of CLOSE.
                    keep = min(len(self.CLOSE) - 1, len(self._buf))
                    self.bytes_dropped += len(self._buf) - keep
                    self._buf = self._buf[-keep:] if keep else ""
                    break
                self.bytes_dropped += idx + len(self.CLOSE)
                self._buf = self._buf[idx + len(self.CLOSE):]
                self._suppressing = False
                continue
            idx = self._buf.find(self.OPEN)
            if idx == -1:
                # No open tag in buffer. Emit safe prefix, hold any trailing
                # chars that could be the start of OPEN.
                safe = len(self._buf)
                for hold in range(min(self._LOOKAHEAD - 1, len(self._buf)), 0, -1):
                    if self.OPEN.startswith(self._buf[-hold:]):
                        safe = len(self._buf) - hold
                        break
                if safe:
                    out.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                break
            out.append(self._buf[:idx])
            self._buf = self._buf[idx + len(self.OPEN):]
            self._suppressing = True
        return "".join(out)

    def flush(self) -> str:
        """Flush at end-of-stream. Drops any unterminated suppressed block."""
        if self._suppressing:
            self.bytes_dropped += len(self._buf)
            self._buf = ""
            self._suppressing = False
            return ""
        out, self._buf = self._buf, ""
        return out


# ---------- types ----------

ToolHandler = Callable[[dict], Awaitable[str]]
"""Async function that executes a single tool call. Receives the parsed
arguments dict. Returns the string content for the resulting `tool` message
the model will see on the next turn. Should not raise; on error, return a
string starting with 'ERROR:'."""


class ToolLabels(NamedTuple):
    """User-facing copy for status chunks emitted around a tool call.

    `start`     — shown once when the tool starts (e.g. "Generating image")
    `progress`  — shown every keepalive_interval_s while the tool runs
                  (e.g. "still generating"). Spark appends "(Ns)" itself.
    """
    start: str
    progress: str


# ---------- helpers ----------

def _make_content_chunk(text: str, served_name: str) -> bytes:
    """Build an OpenAI-shaped SSE chunk that injects assistant content.

    Used for synthetic chunks Spark emits itself (status, progress, error
    notices) — distinct from passthrough chunks that come straight from
    the engine.
    """
    evt = {
        "id": "spark-" + uuid.uuid4().hex[:12],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": served_name,
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": None,
        }],
    }
    return b"data: " + json.dumps(evt).encode() + b"\n\n"


async def _periodic_progress(
    queue: asyncio.Queue,
    served_name: str,
    label: str,
    interval_s: float,
) -> None:
    """Emit a status chunk every `interval_s` seconds while a tool runs.
    Doubles as SSE keepalive. Cancelled by the caller when the tool returns.
    """
    elapsed = 0
    try:
        while True:
            await asyncio.sleep(interval_s)
            elapsed += int(interval_s)
            await queue.put(_make_content_chunk(
                f"_…{label} ({elapsed}s)…_\n", served_name
            ))
    except asyncio.CancelledError:
        return


async def _stream_one_turn(
    client: httpx.AsyncClient, body: dict, queue: asyncio.Queue
) -> list[dict]:
    """Stream one engine turn into `queue`. Return any accumulated tool calls.

    Wire format (OpenAI-compatible, as emitted by SGLang / vLLM):
        data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}
        data: {"choices":[{"delta":{"tool_calls":[
            {"index":0,"id":"...","function":{"name":"...","arguments":""}}
        ]}}]}
        data: {"choices":[{"delta":{"tool_calls":[
            {"index":0,"function":{"arguments":"par"}}
        ]}}]}
        ...
        data: {"choices":[{"finish_reason":"tool_calls"}]}
        data: [DONE]

    Decision per event:
      - delta.tool_calls present → accumulate by index, do NOT forward
      - delta.content / reasoning_content / role → forward verbatim
      - finish_reason="tool_calls" → stop, return calls
      - finish_reason=other → forward, return []
      - [DONE] → engine signals end; runner emits the outer DONE
    """
    stream_body = dict(body)
    stream_body["stream"] = True

    tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    stripper = _ToolCallStripper()

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
                        if finish_reason is None:
                            finish_reason = "stop"
                        continue
                    try:
                        evt = json.loads(payload)
                    except Exception:
                        await queue.put(b"data: " + payload + b"\n\n")
                        continue

                    choices = evt.get("choices") or []
                    if not choices:
                        await queue.put(b"data: " + payload + b"\n\n")
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    fr = choice.get("finish_reason")
                    delta_tools = delta.get("tool_calls")

                    if delta_tools:
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
                    elif delta.get("content") is not None:
                        # Run content through the tool-call stripper. If the
                        # cleaned chunk is empty, drop the whole event so we
                        # don't emit empty deltas downstream.
                        cleaned = stripper.feed(delta["content"])
                        if cleaned:
                            evt["choices"][0]["delta"]["content"] = cleaned
                            await queue.put(
                                b"data: " + json.dumps(evt).encode() + b"\n\n"
                            )
                    elif (
                        delta.get("reasoning_content") is not None
                        or "role" in delta
                    ):
                        await queue.put(b"data: " + payload + b"\n\n")
                    elif fr is not None and fr != "tool_calls":
                        # Flush any held content from the stripper before the
                        # final non-tool-calls finish chunk so trailing safe
                        # text isn't dropped.
                        tail = stripper.flush()
                        if tail:
                            tail_evt = {
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": tail},
                                    "finish_reason": None,
                                }]
                            }
                            await queue.put(
                                b"data: " + json.dumps(tail_evt).encode() + b"\n\n"
                            )
                        await queue.put(b"data: " + payload + b"\n\n")
                    # else: tool_calls finish chunk — drop, runner handles it

                    if fr is not None:
                        finish_reason = fr
    except httpx.HTTPError as e:
        err = json.dumps({"error": {"message": str(e), "type": "upstream_error"}})
        await queue.put(b"data: " + err.encode() + b"\n\n")
        return []

    if finish_reason == "tool_calls" and tool_calls:
        return [tool_calls[i] for i in sorted(tool_calls.keys())]
    return []


# ---------- public API ----------

async def run_agent_stream(
    *,
    client: httpx.AsyncClient,
    body: dict,
    tool_handlers: dict[str, ToolHandler],
    tool_labels: dict[str, ToolLabels] | None = None,
    served_name: str = "agent",
    max_turns: int = 4,
    keepalive_interval_s: float = 15.0,
    log: Callable[[str], None] = _default_log,
    on_between_turns: Callable[[list[dict]], Awaitable[list[dict]]] | None = None,
    on_turn_complete: Callable[[list[dict]], None] | None = None,
):
    """Drive a streaming agent loop until the model produces a turn without
    any tool call (success), or `max_turns` rounds elapse (safety cap).

    Each turn:
      1. Stream the engine response. Forward content/reasoning chunks to
         the client live; accumulate any tool_call deltas internally.
      2. If the turn ended with tool_calls, dispatch each call to its
         handler, append the assistant tool_call message + the tool
         result message to body["messages"], and loop.
      3. If the turn ended without tool_calls, break and return.

    Yields raw SSE bytes ready to write to the client. The final chunk is
    always `data: [DONE]\\n\\n`. On error, an error envelope chunk is
    emitted before [DONE].

    Args:
        client: Pre-configured httpx.AsyncClient pointing at the engine.
        body: OpenAI-shaped chat completion request. Mutated in place
              across turns (assistant + tool messages appended).
        tool_handlers: Map of tool name → async handler(args) -> str.
        tool_labels: Optional UX strings per tool. Defaults to a generic
            "Calling {name}" / "still working".
        served_name: Model name to advertise in synthetic chunks.
        max_turns: Hard cap on agent iterations. Must be ≥ 1.
        keepalive_interval_s: Seconds between progress chunks during a
            tool execution. Doubles as SSE keepalive interval.
        log: Function called with one structured log line per turn.
            Defaults to print(); pass a logger.info wrapper if preferred.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    tool_labels = tool_labels or {}
    queue: asyncio.Queue = asyncio.Queue()

    async def runner() -> None:
        try:
            for turn in range(max_turns):
                # Between-turn hook (e.g. cortex compaction). Only fires
                # on turns 2+, since turn 1 hasn't yet appended anything.
                if turn > 0 and on_between_turns is not None:
                    body["messages"] = await on_between_turns(body["messages"])

                t0 = time.time()
                tool_calls = await _stream_one_turn(client, body, queue)
                dt = time.time() - t0
                if not tool_calls:
                    log(f"[spark] turn {turn+1}: stream done in {dt:.1f}s, no tool calls")
                    if on_turn_complete is not None:
                        try:
                            on_turn_complete(body["messages"])
                        except Exception as e:  # noqa: BLE001
                            log(f"[spark] on_turn_complete hook raised: {e}")
                    return
                names = ",".join(
                    (tc.get("function") or {}).get("name", "?") for tc in tool_calls
                )
                log(f"[spark] turn {turn+1}: stream done in {dt:.1f}s, tool_calls=[{names}]")

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
                    handler = tool_handlers.get(tname)
                    if handler is None:
                        body["messages"].append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "name": tname,
                            "content": f"ERROR: tool '{tname}' is not registered.",
                        })
                        continue

                    labels = tool_labels.get(tname) or ToolLabels(
                        start=f"Calling {tname}",
                        progress="still working",
                    )
                    await queue.put(_make_content_chunk(
                        f"\n\n_{labels.start}…_\n\n", served_name
                    ))
                    progress_task = asyncio.create_task(_periodic_progress(
                        queue, served_name, labels.progress, keepalive_interval_s
                    ))
                    t_tool = time.time()
                    try:
                        content = await handler(args)
                    except Exception as e:
                        content = f"ERROR: tool '{tname}' raised: {e}"
                    finally:
                        progress_task.cancel()
                        try:
                            await progress_task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001
                            pass
                    log(f"[spark] turn {turn+1}: {tname} took "
                        f"{time.time()-t_tool:.1f}s, result_len={len(content)}")

                    body["messages"].append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": tname,
                        "content": content,
                    })
            else:
                log(f"[spark] hit max_turns={max_turns}, stopping")
                await queue.put(_make_content_chunk(
                    "\n\n_Reached the per-turn tool-call limit. Stopping here._\n",
                    served_name,
                ))
        except Exception as e:  # noqa: BLE001
            log(f"[spark] runner unexpected error: {e}")
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


async def run_agent_sync(
    *,
    client: httpx.AsyncClient,
    body: dict,
    tool_handlers: dict[str, ToolHandler],
    max_turns: int = 4,
    log: Callable[[str], None] = _default_log,
    on_between_turns: Callable[[list[dict]], Awaitable[list[dict]]] | None = None,
    on_turn_complete: Callable[[list[dict]], None] | None = None,
) -> dict:
    """Non-streaming twin of run_agent_stream.

    Used for /v1/chat/completions when the caller passed `stream: false`.
    Drives the same agent loop synchronously and returns the final
    OpenAI-shape chat completion JSON.

    The streaming variant is preferred for user-facing UX; this exists
    for API consumers (SDKs, webhooks) that explicitly opted out.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    last_data: dict | None = None
    sync_body = dict(body)
    sync_body["stream"] = False

    for turn in range(max_turns):
        if turn > 0 and on_between_turns is not None:
            sync_body["messages"] = await on_between_turns(sync_body["messages"])

        t0 = time.time()
        try:
            resp = await client.post("/v1/chat/completions", json=sync_body)
        except httpx.HTTPError as e:
            return {"error": {"message": f"upstream error: {e}", "type": "upstream_error"}}
        if resp.status_code != 200:
            return resp.json() if resp.content else {"error": {"message": "engine error"}}

        data = resp.json()
        last_data = data
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []
        proxy_calls = [
            t for t in tool_calls
            if (t.get("function") or {}).get("name") in tool_handlers
        ]
        log(f"[spark/sync] turn {turn+1}: {time.time()-t0:.1f}s, "
            f"tool_calls=[{','.join((tc.get('function') or {}).get('name','?') for tc in proxy_calls) or '-'}]")
        if not proxy_calls:
            if on_turn_complete is not None:
                try:
                    on_turn_complete(sync_body["messages"])
                except Exception as e:  # noqa: BLE001
                    log(f"[spark/sync] on_turn_complete hook raised: {e}")
            return data

        sync_body["messages"].append(msg)
        for tc in proxy_calls:
            fn = tc.get("function") or {}
            tname = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            handler = tool_handlers[tname]
            try:
                content = await handler(args)
            except Exception as e:
                content = f"ERROR: tool '{tname}' raised: {e}"
            sync_body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": tname,
                "content": content,
            })

    log(f"[spark/sync] hit max_turns={max_turns}")
    return last_data or {}
