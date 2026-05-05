"""Sohn LLM translator: chunk + retrieval + memory -> structured TranslationResult."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from .config import Config
from .schema import Chunk, Retrieval, TranslationResult

log = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "prompts"
_FORMULA_PATH = _PROMPT_DIR / "unblinded_formula.md"
_OUTPUT_CONTRACT_PATH = _PROMPT_DIR / "output_contract.md"

_TEMPERATURE = 0.4
_INVALID_RESPONSE_TRUNC = 1000


class TranslationError(RuntimeError):
    """Raised when the LLM fails to return a schema-valid JSON after retries."""


class Translator:
    """Async wrapper around the Sohn OpenAI-compatible chat completions endpoint."""

    _formula_prompt: str = _FORMULA_PATH.read_text(encoding="utf-8")
    _output_contract: str = _OUTPUT_CONTRACT_PATH.read_text(encoding="utf-8")

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        # Per-event timeouts: read=stream-stall budget so a model that stops
        # producing tokens mid-generation fails fast instead of holding the
        # connection open for llm_request_timeout_s.
        timeout = httpx.Timeout(
            connect=15.0,
            read=cfg.llm_stream_read_timeout_s,
            write=15.0,
            pool=15.0,
        )
        self._client = AsyncOpenAI(
            api_key=cfg.sohn_api_key,
            base_url=cfg.sohn_base_url,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def translate(
        self, chunk: Chunk, retrieval: Retrieval, memory_block: str
    ) -> TranslationResult:
        user_content = self._build_user_content(chunk, retrieval, memory_block)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._formula_prompt},
            {"role": "user", "content": user_content},
        ]

        max_attempts = max(1, self._cfg.llm_max_retries + 1)
        last_invalid_text = ""
        last_validation_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            start = time.perf_counter()
            stream = await self._client.chat.completions.create(
                model=self._cfg.sohn_model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self._cfg.sohn_max_tokens,
                temperature=_TEMPERATURE,
                response_format={"type": "json_object"},
                # Sohn-specific: disable chain-of-thought so the model emits the
                # JSON directly instead of routing through reasoning_content.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                # Streaming makes mid-generation stalls fail fast: httpx.Timeout
                # read=stream_read_timeout fires when no token arrives in window.
                stream=True,
                stream_options={"include_usage": True},
            )
            content, prompt_tokens, completion_tokens, finish_reason = (
                await _consume_stream(stream)
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            stripped = _strip_markdown_fences(content)
            try:
                parsed: Any = json.loads(stripped)
                _salvage_fused_seans_fields(parsed)
                result = TranslationResult.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as ex:
                last_invalid_text = content
                last_validation_error = ex
                log.warning(
                    "translate attempt=%d invalid response chunk=%d err=%s",
                    attempt,
                    chunk.chunk_index,
                    ex,
                )
                if attempt >= max_attempts:
                    break
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON matching the "
                            f"schema. Error: {ex}. Return ONLY the JSON object."
                        ),
                    }
                )
                continue

            log.info(
                "translate done chunk=%d rows=%d prompt_tokens=%s completion_tokens=%s "
                "latency_ms=%d finish=%s",
                chunk.chunk_index,
                len(result.rows),
                prompt_tokens,
                completion_tokens,
                latency_ms,
                finish_reason,
            )
            return result

        truncated = last_invalid_text[:_INVALID_RESPONSE_TRUNC]
        raise TranslationError(
            f"LLM failed schema validation after {max_attempts} attempts. "
            f"Last error: {last_validation_error}. Last response (truncated): {truncated}"
        )

    def _build_user_content(
        self, chunk: Chunk, retrieval: Retrieval, memory_block: str
    ) -> str:
        retrieval_block = retrieval.render_for_prompt(
            max_chars_per_doc=self._cfg.retrieval_max_chars_per_doc
        )
        sections = [
            "<output_contract>",
            self._output_contract,
            "</output_contract>",
            "",
            "## Document Processing Context",
            f"Section {chunk.chunk_index} of {chunk.total_chunks} — {chunk.position_context}",
        ]
        if memory_block:
            sections.extend(["", memory_block])
        if retrieval_block:
            sections.extend(["", retrieval_block])
        sections.extend(
            [
                "",
                "## Source content to translate",
                chunk.text,
            ]
        )
        return "\n".join(sections)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_FUSED_SEANS_RE = re.compile(
    r"\n\s*seans?[_ ]approach\s*[:\-]\s*\n?(.*)$", re.DOTALL | re.IGNORECASE
)


def _strip_markdown_fences(content: str) -> str:
    """Some gateways re-wrap json_object output in fences; pre-strip to skip a retry."""
    m = _FENCE_RE.match(content)
    return m.group(1) if m else content


def _salvage_fused_seans_fields(parsed: Any) -> None:
    """In-place repair: model sometimes fuses seans_approach text into seans_processing.

    Detected pattern: row has seans_processing ending with a literal
    `seans_approach:` heading (with optional content after), and seans_approach
    is missing or empty. We split on the heading and lift the tail into
    seans_approach. If a row already has both fields populated, do nothing.
    """
    if not isinstance(parsed, dict):
        return
    rows = parsed.get("rows")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        approach = row.get("seans_approach")
        proc = row.get("seans_processing")
        if not isinstance(proc, str):
            continue
        if isinstance(approach, str) and approach.strip():
            continue  # already populated, leave alone
        m = _FUSED_SEANS_RE.search(proc)
        if not m:
            # missing seans_approach AND no fused header — set empty to satisfy schema
            row["seans_approach"] = ""
            continue
        # split: keep prefix as seans_processing, lift tail into seans_approach
        row["seans_processing"] = proc[: m.start()].rstrip()
        row["seans_approach"] = m.group(1).strip()


async def _consume_stream(
    stream: Any,
) -> tuple[str, int | None, int | None, str | None]:
    """Drain an OpenAI-SDK chat-completion stream into (content, prompt_tokens,
    completion_tokens, finish_reason).

    Streaming gives us a heartbeat: httpx.Timeout(read=N) on the underlying
    client fires if no chunk arrives in N seconds, instead of hanging until the
    full request timeout when the model stalls.
    """
    parts: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    async for chunk in stream:
        # Some chunks (especially the final usage-only chunk) have empty choices.
        choices = getattr(chunk, "choices", None) or []
        for ch in choices:
            delta = getattr(ch, "delta", None)
            text = getattr(delta, "content", None) if delta is not None else None
            if isinstance(text, str) and text:
                parts.append(text)
            fr = getattr(ch, "finish_reason", None)
            if fr:
                finish_reason = fr
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
            completion_tokens = getattr(usage, "completion_tokens", completion_tokens)
    content = "".join(parts)
    if not content:
        raise TranslationError("LLM stream produced no content")
    return content, prompt_tokens, completion_tokens, finish_reason
