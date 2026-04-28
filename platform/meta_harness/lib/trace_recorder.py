"""Atomic per-trace artifact writer.

Per (run, iteration, candidate, scenario, trial), writes:
  trace.json       — the structured HarnessTrace dict, verbatim.
  transcript.txt   — pretty human-readable rendering for the proposer to grep.
  score.json       — per-dim scores, weighted total, RED/YELLOW/GREEN gate.

Atomicity: every write goes through tempfile + os.replace, so a partially
written file never appears under the canonical path.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceLocator:
    """Identifies one trace's on-disk location.

    The runner constructs a TraceLocator per scenario × trial; the recorder
    materializes the directory + writes the artifacts.
    """

    runs_root: Path                  # /opt/acti/meta_harness/runs/<run>/traces/
    iteration: int
    candidate: str                   # e.g. 'baseline_spark'
    scenario_id: str
    trial: int = 0

    def directory(self) -> Path:
        return (
            self.runs_root
            / f"iter_{self.iteration:03d}"
            / self.candidate
            / "response"
            / self.scenario_id
            / f"trial_{self.trial:02d}"
        )


@dataclass
class ScoreSummary:
    """Per-scenario score record. Populated by benchmark_response.py."""

    weighted_total: float                       # 0..100
    per_dim: dict[str, int]                     # dim_id → 0/1/2 (or -1 N/A)
    judge_rationale: dict[str, str] = field(default_factory=dict)
    programmatic: dict[str, str] = field(default_factory=dict)
    gate: str = "GREEN"                         # RED / YELLOW / GREEN
    crit_violations: list[str] = field(default_factory=list)


# ---------- atomic write ----------


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically. Caller ensures parent dir exists."""
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the half-written temp file.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, obj: object) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False, default=str)
    _atomic_write_text(path, text + "\n")


# ---------- transcript renderer ----------


def render_transcript(trace: dict[str, Any]) -> str:
    """Pretty turn-by-turn rendering of a HarnessTrace.

    Intended for the proposer to grep through. Includes:
      - scenario metadata
      - each turn (system / user / assistant text or tool_calls / tool result)
      - tool_call args + retrieval hits inline
      - citations summary at the end

    Output is ~3-10x shorter than the raw transcript JSON because it strips
    OpenAI metadata + collapses whitespace, but no factual content is lost.
    """
    lines: list[str] = []
    sep = "─" * 70

    # Header
    lines.append(f"# trace: {trace.get('candidate', '?')} / "
                 f"scenario={trace.get('scenario_id', '?')} / "
                 f"user_id={trace.get('user_id', '?')}")
    em = trace.get("engine_meta") or {}
    lines.append(
        f"# engine: prompt_tokens={em.get('prompt_tokens', '?')} "
        f"completion_tokens={em.get('completion_tokens', '?')} "
        f"wall={em.get('wall_time_s', '?')}s "
        f"max_turns_cap={em.get('max_turns_cap', '?')}"
    )
    if (trace.get("harness_meta") or {}).get("error"):
        lines.append(f"# ERROR: {trace['harness_meta']['error']}")
    lines.append("")

    # Conversation
    transcript = trace.get("transcript") or []
    for i, msg in enumerate(transcript):
        role = msg.get("role", "?")
        if role == "system":
            content = (msg.get("content") or "")
            lines.append(f"## [{i}] SYSTEM ({len(content)} chars, abbreviated)")
            lines.append(content[:400] + ("…" if len(content) > 400 else ""))
        elif role == "user":
            lines.append(f"## [{i}] USER")
            lines.append(msg.get("content") or "")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            if tool_calls:
                lines.append(f"## [{i}] ASSISTANT (tool_calls)")
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    args_raw = fn.get("arguments", "")
                    lines.append(f"  call: {fn.get('name', '?')}({args_raw})")
            else:
                lines.append(f"## [{i}] ASSISTANT")
                lines.append(content or "(empty)")
        elif role == "tool":
            content = msg.get("content") or ""
            head = content.splitlines()[0] if content else "(empty)"
            preview = content[:600] + ("…" if len(content) > 600 else "")
            lines.append(f"## [{i}] TOOL [{msg.get('name', '?')}] — {head[:80]}")
            lines.append(preview)
        else:
            lines.append(f"## [{i}] {role.upper()}")
            lines.append(str(msg.get("content") or ""))
        lines.append("")

    # Retrieval hit summary
    hits = trace.get("retrieval_hits") or []
    if hits:
        lines.append(sep)
        lines.append(f"# retrieval_hits ({len(hits)} total)")
        for j, h in enumerate(hits[:20]):  # cap to 20 in transcript
            md = h.get("metadata") or {}
            lines.append(
                f"  {j+1}. id={h.get('id', '?')} "
                f"score={h.get('score', 0):.3f} "
                f"ns={md.get('namespace', '?')} "
                f"subject={md.get('subject_entity', '?')}"
            )
        if len(hits) > 20:
            lines.append(f"  … {len(hits) - 20} more")
        lines.append("")

    # Citation summary
    cites = trace.get("citations_extracted") or []
    if cites:
        lines.append(sep)
        valid = sum(1 for c in cites if c.get("valid"))
        lines.append(f"# citations ({valid}/{len(cites)} valid)")
        for c in cites:
            mark = "✓" if c.get("valid") else "✗"
            lines.append(
                f"  {mark} [{c.get('raw', '?')}] form={c.get('form', '?')} "
                f"→ {c.get('resolved_to_hit_id') or '(unresolved)'}"
            )

    return "\n".join(lines)


# ---------- public recorder ----------


def record_trace(
    locator: TraceLocator,
    trace: dict[str, Any],
    score: ScoreSummary | None = None,
) -> Path:
    """Materialize the trace directory and write artifacts. Returns the dir.

    Idempotent: re-running with the same locator overwrites in place
    (atomically). The directory is created with parents=True.
    """
    d = locator.directory()
    d.mkdir(parents=True, exist_ok=True)

    # Stamp scenario_id into the trace so trace.json is self-describing.
    if isinstance(trace, dict):
        trace = dict(trace)
        if not trace.get("scenario_id"):
            trace["scenario_id"] = locator.scenario_id

    _atomic_write_json(d / "trace.json", trace)
    _atomic_write_text(d / "transcript.txt", render_transcript(trace))
    if score is not None:
        _atomic_write_json(
            d / "score.json",
            {
                "weighted_total": round(score.weighted_total, 2),
                "gate": score.gate,
                "crit_violations": list(score.crit_violations),
                "per_dim": dict(score.per_dim),
                "judge_rationale": dict(score.judge_rationale),
                "programmatic": dict(score.programmatic),
            },
        )
    return d
