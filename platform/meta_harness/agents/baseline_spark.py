"""baseline_spark — Terminus-2 analog: faithful wrapper around production Spark.

This is the parent candidate every evolutionary candidate descends from.
It must reproduce the existing 94.4/100 eval baseline within ±0.5 to be
considered a valid baseline (acceptance test in tests/).

What it does (in order, per `run`):
  1. Build the OpenAI-shape request body with system prompt + user query
     + the recall_context tool schema advertised directly (no load_tool
     indirection — same as platform/eval/lib/runner.py).
  2. Wrap the production library.handle_recall_context handler so we can
     capture the raw retrieval hits (the handler upstream returns a
     formatted string; we need the structured hits for the trace).
  3. Drive the production spark.run_agent_sync loop with max_turns=6.
  4. Assemble a HarnessTrace from body.messages, the captured hits,
     and the engine response's usage block.

What it deliberately does NOT do:
  - Cortex memory injection / compaction. The eval baseline doesn't use
    cortex, so neither does this. A separate baseline_spark_cortex
    candidate (out of scope for v1) would add it.
  - Streaming. We use spark.run_agent_sync because the harness needs the
    final OpenAI response object whole; the trace doesn't need SSE chunks.
  - Identity pre-filter, citation post-validation, attribution recovery,
    or any other clever harness logic. Those are search-axes for proposed
    candidates; this baseline is the unmodified production loop.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# ---------- import production spark + library ----------
#
# The harness lives under platform/meta_harness/; the production code lives
# under platform/proxy/. On the pod, the equivalent layout is
# /opt/acti/meta_harness/ and /opt/acti/proxy/. Insert the proxy dir onto
# sys.path so we can import the loop code without packaging.

_THIS_DIR = Path(__file__).resolve().parent
_PROXY_DIR_CANDIDATES = [
    _THIS_DIR.parent.parent / "proxy",     # repo: platform/proxy/
    Path("/opt/acti/proxy"),               # pod
    Path(os.environ.get("ACTI_PROXY_DIR", "/nonexistent")),
]
for _p in _PROXY_DIR_CANDIDATES:
    if _p.is_dir() and (_p / "spark.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
else:
    raise ImportError(
        f"could not find production spark.py in any of {_PROXY_DIR_CANDIDATES}; "
        f"set ACTI_PROXY_DIR= or check repo layout"
    )

import library  # type: ignore[import-not-found]   # noqa: E402
import spark  # type: ignore[import-not-found]   # noqa: E402

# ---------- harness lib (siblings) ----------

sys.path.insert(0, str(_THIS_DIR.parent))
from lib.citations import validate_response_citations  # type: ignore[import-not-found]   # noqa: E402
from lib.harness import (  # type: ignore[import-not-found]   # noqa: E402
    CitationTrace,
    HarnessTrace,
    ToolCallTrace,
)
from lib.pod_clients import make_engine_client  # type: ignore[import-not-found]   # noqa: E402

# ---------- defaults ----------

DEFAULT_SYSTEM_PROMPT_PATH = Path(
    os.environ.get(
        "ACTI_SYSTEM_PROMPT_PATH",
        str(_THIS_DIR.parent.parent / "system_prompts" / "sohn.txt"),
    )
)
DEFAULT_MAX_TURNS = int(os.environ.get("ACTI_HARNESS_MAX_TURNS", "6"))
DEFAULT_SERVED_NAME = os.environ.get("ACTI_HARNESS_SERVED_NAME", "Sohn")


def _read_system_prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"system prompt not found at {path}; set ACTI_SYSTEM_PROMPT_PATH"
        ) from e


# ---------- the candidate class ----------


class BaselineSparkHarness:
    """Faithful production-Spark harness. Implements ActiHarness Protocol."""

    name: str = "baseline_spark"

    def __init__(
        self,
        *,
        engine_base_url: str,
        retrieval_base_url: str,
        api_key: str,
        retrieval_api_key: str,
        system_prompt_path: str | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        served_name: str = DEFAULT_SERVED_NAME,
        **_kwargs: Any,
    ) -> None:
        self._engine_base_url = engine_base_url
        self._engine_api_key = api_key
        self._max_turns = max_turns
        self._served_name = served_name

        # Production library reads its base URL + key from env. Ensure they're
        # set per-instance so concurrent harnesses don't stomp each other.
        os.environ["ACTI_LIBRARY_BASE_URL"] = retrieval_base_url
        os.environ["ACTI_LIBRARY_API_KEY"] = retrieval_api_key
        # Force-reload the module-level cache in library.py.
        library.LIBRARY_BASE_URL = retrieval_base_url.rstrip("/")
        library.LIBRARY_API_KEY = retrieval_api_key

        self._system_prompt = _read_system_prompt(
            Path(system_prompt_path) if system_prompt_path
            else DEFAULT_SYSTEM_PROMPT_PATH
        )

    # ---------- the contract ----------

    async def run(
        self, user_query: str, user_id: str
    ) -> tuple[str, HarnessTrace]:
        """Drive one scenario through the production Spark loop.

        See module docstring for what's captured. Never raises; returns
        an ERROR-prefixed response + an error-tagged trace on failure.
        """
        scenario_meta = {"user_id": user_id, "candidate": self.name}
        captured_calls: list[ToolCallTrace] = []
        captured_hits_flat: list[dict] = []
        captured_retrieval_meta: dict[str, Any] = {}

        body = self._build_body(user_query)
        tool_handlers = {
            "recall_context": self._make_capturing_handler(
                captured_calls, captured_hits_flat, captured_retrieval_meta
            ),
        }

        t0 = time.time()
        try:
            async with make_engine_client(
                base_url=self._engine_base_url,
                api_key=self._engine_api_key,
            ) as client:
                result = await spark.run_agent_sync(
                    client=client,
                    body=body,
                    tool_handlers=tool_handlers,
                    max_turns=self._max_turns,
                    log=lambda line: None,  # silence — runner controls logging
                )
        except Exception as e:  # noqa: BLE001
            wall = time.time() - t0
            return f"ERROR: harness raised: {e}", self._error_trace(
                user_query, scenario_meta, str(e), wall
            )

        wall = time.time() - t0
        final_response, engine_meta = self._extract_final_response(result, wall)

        # Citation validation (programmatic; no judge needed).
        citations = validate_response_citations(final_response, captured_hits_flat)
        citation_traces: list[CitationTrace] = [
            {
                "raw": c.raw,
                "form": c.form,
                "resolved_to_hit_id": c.resolved_to_hit_id,
                "valid": c.valid,
            }
            for c in citations
        ]

        trace: HarnessTrace = {
            "candidate": self.name,
            "scenario_id": "",  # runner sets this
            "user_id": user_id,
            "user_query": user_query,
            "final_response": final_response,
            "transcript": list(body["messages"]),
            "tool_calls": captured_calls,
            "retrieval_hits": captured_hits_flat,
            "retrieval_meta": captured_retrieval_meta,
            "cortex_deltas": [],  # baseline doesn't use cortex
            "engine_meta": engine_meta,
            "citations_extracted": citation_traces,
            "harness_meta": {
                "version": "v1",
                "policy": "production-spark-direct-tool",
                "max_turns_cap": self._max_turns,
            },
        }
        return final_response, trace

    # ---------- private helpers ----------

    def _build_body(self, user_query: str) -> dict[str, Any]:
        """OpenAI-shape body for spark.run_agent_sync.

        Mirrors platform/eval/lib/runner.py — system prompt + user message
        + recall_context tool schema advertised directly.
        """
        return {
            "model": self._served_name,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_query},
            ],
            "tools": [library.RECALL_CONTEXT_TOOL],
            "tool_choice": "auto",
            "stream": False,
        }

    def _make_capturing_handler(
        self,
        captured_calls: list[ToolCallTrace],
        captured_hits_flat: list[dict],
        captured_meta: dict[str, Any],
    ):
        """Wrap library.handle_recall_context to capture raw hits + timing.

        We re-implement the validation + retrieve call inline (rather than
        wrapping handle_recall_context) so we can intercept the hits dict
        between `_call_retrieve` and `_format_hits`. Matches production
        behavior token-for-token in the formatted string the model sees.
        """

        async def _handler(args: dict) -> str:
            t0 = time.time()
            turn_idx = sum(
                1 for c in captured_calls if c["name"] == "recall_context"
            ) + 1

            # Mirror library.handle_recall_context's validation.
            if not library.library_enabled():
                msg = "ERROR: the Unblinded library is not configured on this deployment."
                captured_calls.append(self._mk_call_trace(
                    turn_idx, args, msg, [], int((time.time() - t0) * 1000)
                ))
                return msg

            query = (args.get("query") or "").strip()
            if not query:
                msg = "ERROR: `query` is required."
                captured_calls.append(self._mk_call_trace(
                    turn_idx, args, msg, [], int((time.time() - t0) * 1000)
                ))
                return msg

            intent = args.get("intent") or "general"
            if (
                intent in {"person-recall", "continuity-snapshot"}
                and not args.get("subject_entity")
            ):
                msg = (
                    f"ERROR: intent={intent!r} requires `subject_entity` "
                    f"(format: `user:<slug>`, e.g. `user:adam-gugino`). "
                    f"Either provide it, or call with intent=general."
                )
                captured_calls.append(self._mk_call_trace(
                    turn_idx, args, msg, [], int((time.time() - t0) * 1000)
                ))
                return msg
            if intent == "case-lookup" and not args.get("case_id"):
                msg = (
                    f"ERROR: intent='case-lookup' requires `case_id` "
                    f"(format: `cf-<slug>`, e.g. `cf-denver-family-law`). "
                    f"Either provide it, or call with intent=general."
                )
                captured_calls.append(self._mk_call_trace(
                    turn_idx, args, msg, [], int((time.time() - t0) * 1000)
                ))
                return msg

            payload: dict[str, Any] = {"query": query, "intent": intent}
            if args.get("subject_entity"):
                payload["subject_entity"] = args["subject_entity"]
            if args.get("case_id"):
                payload["case_id"] = args["case_id"]
            if args.get("top_k"):
                payload["top_k"] = max(
                    1, min(int(args["top_k"]), library.LIBRARY_MAX_HITS)
                )
            else:
                payload["top_k"] = library.LIBRARY_MAX_HITS

            result = await library._call_retrieve(payload)
            duration_ms = int((time.time() - t0) * 1000)

            if "error" in result:
                msg = f"ERROR: {result['error']}"
                captured_calls.append(self._mk_call_trace(
                    turn_idx, args, msg, [], duration_ms
                ))
                return msg

            hits = result.get("hits") or []
            captured_hits_flat.extend(hits)
            captured_meta.setdefault("intents_used", []).append(intent)
            captured_meta["last_intent"] = intent
            captured_meta["last_payload"] = payload

            formatted = library._format_hits(intent, query, hits)
            captured_calls.append(
                self._mk_call_trace(turn_idx, args, formatted, hits, duration_ms)
            )
            return formatted

        return _handler

    @staticmethod
    def _mk_call_trace(
        turn: int,
        args_parsed: dict,
        result_text: str,
        result_hits: list[dict],
        duration_ms: int,
    ) -> ToolCallTrace:
        """Assemble a ToolCallTrace from the data we have inside the handler."""
        return {
            "turn": turn,
            "name": "recall_context",
            "arguments_raw": json.dumps(args_parsed, ensure_ascii=False),
            "arguments_sanitized": dict(args_parsed),
            "arguments_parsed": dict(args_parsed),
            "result_text": result_text,
            "result_hits": list(result_hits),
            "duration_ms": duration_ms,
        }

    def _extract_final_response(
        self, result: dict, wall_time_s: float
    ) -> tuple[str, dict]:
        """Pull final_response + engine_meta out of the spark.run_agent_sync result."""
        if not result or "error" in result:
            err = (result or {}).get("error", {}).get("message", "engine error")
            return f"ERROR: {err}", {
                "n_turns": 0,
                "max_turns_used": 0,
                "max_turns_cap": self._max_turns,
                "wall_time_s": round(wall_time_s, 3),
            }
        choices = result.get("choices") or []
        msg = (choices[0].get("message") if choices else {}) or {}
        content = msg.get("content") or ""
        usage = result.get("usage") or {}
        return content, {
            "n_turns": 0,                              # filled by runner if needed
            "max_turns_used": 0,                       # not reported by spark.sync
            "max_turns_cap": self._max_turns,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "wall_time_s": round(wall_time_s, 3),
        }

    def _error_trace(
        self, user_query: str, meta: dict, err: str, wall_time_s: float
    ) -> HarnessTrace:
        return {
            "candidate": self.name,
            "scenario_id": "",
            "user_id": meta["user_id"],
            "user_query": user_query,
            "final_response": f"ERROR: {err}",
            "transcript": [],
            "tool_calls": [],
            "retrieval_hits": [],
            "retrieval_meta": {},
            "cortex_deltas": [],
            "engine_meta": {
                "n_turns": 0,
                "max_turns_used": 0,
                "max_turns_cap": self._max_turns,
                "wall_time_s": round(wall_time_s, 3),
            },
            "citations_extracted": [],
            "harness_meta": {
                "version": "v1",
                "policy": "production-spark-direct-tool",
                "error": err,
            },
        }


# Stanford convention: candidates expose `AgentHarness` at module scope so
# the runner can instantiate via `<module>:AgentHarness`.
AgentHarness = BaselineSparkHarness
