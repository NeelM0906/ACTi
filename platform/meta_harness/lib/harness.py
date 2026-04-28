"""ActiHarness Protocol + trace-schema TypedDict.

Every candidate harness implements ActiHarness. The trace dict it returns
is the load-bearing artifact of Meta-Harness — see ../domain_spec.md for
the full schema rationale.
"""
from __future__ import annotations

from typing import Any, Awaitable, Literal, Protocol, TypedDict


# ---------- trace schema ----------


class ToolCallTrace(TypedDict, total=False):
    """One tool_call entry — captures both raw and parsed args + result."""

    turn: int
    name: str
    arguments_raw: str           # exactly as engine emitted (pre-sanitize)
    arguments_sanitized: dict[str, Any]
    arguments_parsed: dict[str, Any]
    result_text: str
    result_hits: list[dict[str, Any]]  # raw retrieval hits w/ chunk_ids
    duration_ms: int


class CortexDelta(TypedDict, total=False):
    phase: str  # 'before_turn_N' / 'after_turn_N' / 'compaction'
    user_id_partition: str
    size_bytes: int
    added: list[dict[str, Any]]
    compacted: list[dict[str, Any]]


class EngineMeta(TypedDict, total=False):
    n_turns: int
    max_turns_used: int
    max_turns_cap: int
    prompt_tokens: int
    completion_tokens: int
    wall_time_s: float


class RetrievalMeta(TypedDict, total=False):
    intent_used: str
    config_used: dict[str, Any]
    stage_latencies: dict[str, int]
    rerank_skipped: bool
    fallback_used: str | None


CitationForm = Literal["full", "tag-prefix", "source-title", "unknown"]


class CitationTrace(TypedDict, total=False):
    raw: str
    form: CitationForm
    resolved_to_hit_id: str | None
    valid: bool


class HarnessTrace(TypedDict, total=False):
    """The full per-query trace. ALL fields are written by the harness;
    none are filled in by the runner (the runner sets `scenario_id` /
    `user_id` on a wrapper, not inside the trace).
    """

    candidate: str
    scenario_id: str
    user_id: str
    user_query: str
    final_response: str
    transcript: list[dict[str, Any]]      # raw OpenAI-shape messages
    tool_calls: list[ToolCallTrace]
    retrieval_hits: list[dict[str, Any]]  # flattened across all calls
    retrieval_meta: RetrievalMeta
    cortex_deltas: list[CortexDelta]
    engine_meta: EngineMeta
    citations_extracted: list[CitationTrace]
    harness_meta: dict[str, Any]


# ---------- the harness contract ----------


class ActiHarness(Protocol):
    """Every candidate must satisfy this Protocol.

    Implementations live in agents/<name>.py and expose `AgentHarness =
    ConcreteClass` at module scope so the runner can instantiate by
    `<module>:AgentHarness` (Stanford convention).

    Construction takes **kwargs only — proposer-written candidates can
    add extra constructor knobs without breaking the runner.
    """

    name: str  # candidate identifier, e.g. 'baseline_spark'

    def __init__(
        self,
        *,
        engine_base_url: str,
        retrieval_base_url: str,
        api_key: str,
        retrieval_api_key: str,
        **kwargs: Any,
    ) -> None: ...

    def run(
        self, user_query: str, user_id: str
    ) -> Awaitable[tuple[str, HarnessTrace]]:
        """Async. Returns (final_response, trace).

        Must NEVER raise. Internal errors are surfaced inside the trace
        (`harness_meta["error"]`) and the response carries an explicit
        ERROR: prefix. The runner inspects `harness_meta["error"]` to
        decide whether to count the run as a crash for benchmark stats.
        """
        ...
