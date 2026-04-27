"""Benchmark runner — drives Sohn through every scenario, captures everything.

Architecture mirrors the production agent loop (spark.py) but is instrumented
for evaluation: it calls /raw/v1/chat/completions directly with the Sohn
system prompt and the recall_context tool schema advertised, and dispatches
recall_context calls to the retrieval service itself so we can capture the
exact args and hits.

We intentionally bypass the proxy's load_tool indirection here. The eval is
not gating on plumbing; it's gating on the model's behavior given the
persona, the tool schema, and access to the corpus. Every byte the model
sees in production is reproduced here.

Public API:
    run_benchmark(scenarios, rubric, *, sut_base_url, retrieval_base_url, ...) -> RunReport
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import yaml

from . import checks as _checks
from . import judge as _judge


# ---------- configuration ----------

DEFAULT_SUT_BASE_URL = os.environ.get(
    "ACTI_EVAL_SUT_BASE_URL", "http://127.0.0.1:8888/raw/v1"
)
DEFAULT_RETRIEVAL_BASE_URL = os.environ.get(
    "ACTI_LIBRARY_BASE_URL", "https://acti-retrieval-production.up.railway.app"
)
DEFAULT_RUNS_DIR = Path(os.environ.get(
    "ACTI_EVAL_RUNS_DIR", "/opt/acti/eval/runs"
))
DEFAULT_SYSTEM_PROMPT_PATH = Path(os.environ.get(
    "ACTI_SYSTEM_PROMPT_PATH", "/opt/acti/system_prompts/sohn.txt"
))
DEFAULT_MAX_TURNS = 4
DEFAULT_PER_SCENARIO_TIMEOUT_S = 90.0


def _log(msg: str) -> None:
    print(msg, file=sys.stdout, flush=True)


# ---------- recall_context tool schema (minimal client-side mirror) ----------
#
# Source of truth is platform/proxy/library.py — we duplicate the shape here
# so the runner can advertise the tool to the model without importing the
# proxy package.

RECALL_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "recall_context",
        "description": (
            "Search the Unblinded knowledge corpus — Sean Callagy's body of work, "
            "the Unblinded Results Formula, Zone Action and related teachings, dossiers "
            "on people in the ecosystem (Sean, Adam, Kai, and others with `user:*` slugs), "
            "client cases (with `cf-*` slugs like cf-denver-family-law), products, and "
            "structural identity records. Returns ranked text snippets with "
            "`[source_title]` citations you can quote from.\n\n"
            "Choose `intent` based on the question shape:\n"
            "  - `explain-teaching` — \"what is X / explain Y\" (Unblinded concepts, methodology)\n"
            "  - `person-recall` — \"tell me about person P\" (REQUIRES `subject_entity`)\n"
            "  - `case-lookup` — per-case content (REQUIRES `case_id`)\n"
            "  - `kai-memory` — Kai's recent journal\n"
            "  - `continuity-snapshot` — long structural identity dump (REQUIRES `subject_entity`)\n"
            "  - `general` — catch-all when none of the above clearly fits"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "intent": {
                    "type": "string",
                    "enum": [
                        "explain-teaching", "person-recall", "case-lookup",
                        "kai-memory", "continuity-snapshot", "general",
                    ],
                    "default": "general",
                },
                "subject_entity": {"type": "string"},
                "case_id": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}


# ---------- types ----------

@dataclass
class ScenarioRun:
    scenario_id: str
    intent: str
    user_prompt: str
    response_text: str
    tool_calls: list[dict]              # each: {function: {name}, args: {...}, hits: [...]}
    retrieved_hits: list[dict]          # flattened across all recall_context calls
    latency_s: float
    sut_error: Optional[str] = None


@dataclass
class ScenarioReport:
    scenario: dict
    run: ScenarioRun
    programmatic: dict[str, _checks.CheckResult]
    judgement: _judge.JudgeResult
    score_0_100: float                  # weighted normalized
    failed_critical: list[str]          # dim_ids that scored 0 and are CRITICAL


@dataclass
class RunReport:
    run_id: str
    started_at: float
    finished_at: float
    sut_base_url: str
    retrieval_base_url: str
    rubric_version: int
    scenarios: list[ScenarioReport]
    summary: dict                       # aggregate stats
    coverage: Optional[dict] = None     # filled in by coverage.py if requested


# ---------- scenario loading ----------

def load_scenarios(scenarios_dir: Path) -> list[dict]:
    """Load every scenarios/*.yaml under `scenarios_dir`. Returns a flat list."""
    out: list[dict] = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        # Skip macOS AppleDouble metadata files that ride along in tar transfers.
        if path.name.startswith("._"):
            continue
        # Read as utf-8 text first; passing the string side-steps PyYAML's
        # internal byte-decode which uses locale encoding on text streams.
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "scenarios" not in data:
            _log(f"[runner] WARN: {path.name} has no 'scenarios' key, skipped")
            continue
        for s in data["scenarios"]:
            s["_source_file"] = path.name
            out.append(s)
    return out


# ---------- retrieval helper ----------

async def _call_retrieve(
    client: httpx.AsyncClient, base_url: str, api_key: str, args: dict,
) -> dict:
    """POST /v1/retrieve. Returns service JSON or {'error': ...}."""
    payload: dict = {
        "query": args.get("query", "")[:2000],
        "intent": args.get("intent", "general"),
        "embedding_model": "sohn-embedding-sm",
        "top_k": int(args.get("top_k", 8)),
    }
    if args.get("subject_entity"):
        payload["subject_entity"] = args["subject_entity"]
    if args.get("case_id"):
        payload["case_id"] = args["case_id"]

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = await client.post(
            f"{base_url}/v1/retrieve", json=payload, headers=headers, timeout=30.0,
        )
    except httpx.HTTPError as e:
        return {"error": f"retrieval transport: {e}"}
    if r.status_code != 200:
        return {"error": f"retrieval {r.status_code}: {r.text[:300]}"}
    try:
        return r.json()
    except ValueError as e:
        return {"error": f"retrieval non-JSON: {e}"}


def _format_retrieval_for_model(query: str, intent: str, hits: list[dict]) -> str:
    """Mirror library._format_hits so the model sees production-shaped tool results."""
    if not hits:
        return (
            f"No hits in the Unblinded library for query={query!r} intent={intent!r}. "
            f"Either the corpus genuinely lacks this content or the question phrasing "
            f"didn't match well."
        )
    lines = [f"Library returned {len(hits)} hit(s) for query={query!r} intent={intent!r}.",
             "Use these snippets to ground your answer. Cite by source_title where natural.", ""]
    for i, h in enumerate(hits, 1):
        m = h.get("metadata", {}) or {}
        src = m.get("source_title", "(unknown source)")
        ns = m.get("namespace", "?")
        score = h.get("score", 0.0)
        text = (h.get("text") or "").strip()
        if len(text) > 1200:
            text = text[:1200].rstrip() + "…"
        lines.append(f"{i}. [{src}] (ns={ns}, score={score:.2f})")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------- agent loop ----------

async def _drive_one_scenario(
    *,
    scenario: dict,
    system_prompt: str,
    sut_client: httpx.AsyncClient,
    sut_base_url: str,
    sut_api_key: str,
    retrieval_client: httpx.AsyncClient,
    retrieval_base_url: str,
    retrieval_api_key: str,
    max_turns: int,
    timeout_s: float,
) -> ScenarioRun:
    """Drive one scenario end-to-end, capturing tool calls and hits."""
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario["user_prompt"]},
    ]
    body = {
        "model": "Sohn",
        "messages": messages,
        "tools": [RECALL_CONTEXT_TOOL],
        "tool_choice": "auto",
        "stream": False,
        "max_tokens": 1500,
        "temperature": 0.6,
    }
    headers = {"Authorization": f"Bearer {sut_api_key}"} if sut_api_key else {}

    captured_calls: list[dict] = []
    flattened_hits: list[dict] = []
    final_text = ""
    err: Optional[str] = None
    t0 = time.time()
    deadline = t0 + timeout_s

    for turn in range(max_turns):
        if time.time() > deadline:
            err = f"timeout after {timeout_s}s"
            break
        try:
            r = await sut_client.post(
                f"{sut_base_url}/chat/completions",
                json=body, headers=headers, timeout=60.0,
            )
        except httpx.HTTPError as e:
            err = f"SUT transport: {e}"
            break
        if r.status_code != 200:
            err = f"SUT {r.status_code}: {r.text[:300]}"
            break

        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        finish = choice.get("finish_reason")
        tool_calls = msg.get("tool_calls") or []

        if finish == "tool_calls" and tool_calls:
            # Append assistant turn carrying the tool_calls.
            body["messages"].append({
                "role": "assistant", "content": msg.get("content"),
                "tool_calls": tool_calls,
            })
            # Dispatch each call.
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                if name == "recall_context":
                    res = await _call_retrieve(
                        retrieval_client, retrieval_base_url, retrieval_api_key, args,
                    )
                    if "error" in res:
                        tool_text = f"ERROR: {res['error']}"
                        hits: list[dict] = []
                    else:
                        hits = res.get("hits") or []
                        flattened_hits.extend(hits)
                        tool_text = _format_retrieval_for_model(
                            args.get("query", ""), args.get("intent", "general"), hits,
                        )
                    captured_calls.append({
                        "function": {"name": name},
                        "args": args,
                        "hits": hits,
                    })
                else:
                    tool_text = (
                        f"ERROR: tool {name!r} is not handled by the eval runner. "
                        f"Continue without it."
                    )
                    captured_calls.append({
                        "function": {"name": name}, "args": args, "hits": [],
                    })
                body["messages"].append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "content": tool_text,
                })
            continue  # next turn

        # No tool calls — capture content and stop.
        final_text = msg.get("content") or ""
        break
    else:
        err = f"hit max_turns={max_turns} without finishing"

    return ScenarioRun(
        scenario_id=scenario["id"],
        intent=scenario.get("intent", "?"),
        user_prompt=scenario["user_prompt"],
        response_text=final_text,
        tool_calls=captured_calls,
        retrieved_hits=flattened_hits,
        latency_s=time.time() - t0,
        sut_error=err,
    )


# ---------- aggregation ----------

def _critical_dims(rubric: dict) -> set[str]:
    return {d["id"] for d in rubric.get("dimensions", []) if d.get("priority") == "CRITICAL"}


def _weight(rubric: dict, dim_id: str) -> int:
    weights = rubric.get("weights", {})
    by_id = {d["id"]: d for d in rubric.get("dimensions", [])}
    pri = by_id.get(dim_id, {}).get("priority", "MED")
    return int(weights.get(pri, 1))


def _score_one(rubric: dict, judgement: _judge.JudgeResult) -> tuple[float, list[str]]:
    """Weighted normalized 0–100 + list of CRITICAL dim_ids that scored 0."""
    crits = _critical_dims(rubric)
    total_weight = 0
    total_value = 0
    failed: list[str] = []
    for did, ds in judgement.scores.items():
        if ds.score == -1:
            continue
        w = _weight(rubric, did)
        total_weight += w * 2  # max value per dim is 2
        total_value += w * ds.score
        if did in crits and ds.score == 0:
            failed.append(did)
    if total_weight == 0:
        return 0.0, failed
    return round(100.0 * total_value / total_weight, 1), failed


def _summarize(rubric: dict, reports: list[ScenarioReport]) -> dict:
    by_dim: dict[str, list[int]] = {}
    by_intent: dict[str, list[float]] = {}
    crit_failures: dict[str, int] = {}
    for r in reports:
        by_intent.setdefault(r.scenario.get("intent", "?"), []).append(r.score_0_100)
        for did, ds in r.judgement.scores.items():
            if ds.score == -1:
                continue
            by_dim.setdefault(did, []).append(ds.score)
        for did in r.failed_critical:
            crit_failures[did] = crit_failures.get(did, 0) + 1
    dim_means = {did: round(statistics.mean(vs), 2) for did, vs in by_dim.items()}
    intent_means = {iv: round(statistics.mean(vs), 1) for iv, vs in by_intent.items()}
    overall = round(statistics.mean([r.score_0_100 for r in reports]), 1) if reports else 0.0

    # Gating
    gates = rubric.get("gates", {})
    warn_threshold = float(gates.get("warn_if_high_dim_avg_below", 1.5))
    high_dims = {d["id"] for d in rubric.get("dimensions", []) if d.get("priority") == "HIGH"}
    high_dim_warnings = [
        did for did, mean in dim_means.items()
        if did in high_dims and mean < warn_threshold
    ]
    overall_status = "GREEN"
    if crit_failures:
        overall_status = "RED"
    elif high_dim_warnings:
        overall_status = "YELLOW"

    return {
        "overall_score_0_100": overall,
        "overall_status": overall_status,
        "n_scenarios": len(reports),
        "dim_means": dim_means,
        "intent_means": intent_means,
        "critical_failures_by_dim": crit_failures,
        "high_dim_warnings": high_dim_warnings,
    }


# ---------- entry point ----------

async def run_benchmark(
    *,
    scenarios: list[dict],
    rubric: dict,
    system_prompt: str,
    sut_base_url: str = DEFAULT_SUT_BASE_URL,
    sut_api_key: Optional[str] = None,
    retrieval_base_url: str = DEFAULT_RETRIEVAL_BASE_URL,
    retrieval_api_key: Optional[str] = None,
    judge_base_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    judge_cache_dir: Optional[Path] = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    per_scenario_timeout_s: float = DEFAULT_PER_SCENARIO_TIMEOUT_S,
    concurrency: int = 4,
    use_judge: bool = True,
    use_judge_cache: bool = True,
) -> RunReport:
    """Run every scenario, score with programmatic + judge, return RunReport.

    `scenarios` and `rubric` are dicts loaded by the caller.
    `system_prompt` is the verbatim sohn.txt content.
    """
    sut_api_key = sut_api_key or os.environ.get("ACTI_EVAL_API_KEY", "")
    retrieval_api_key = retrieval_api_key or os.environ.get("ACTI_LIBRARY_API_KEY", "")
    judge_api_key = judge_api_key or sut_api_key
    judge_base_url = judge_base_url or sut_base_url
    judge_cache_dir = judge_cache_dir or _judge.DEFAULT_CACHE_DIR

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    started = time.time()
    _log(f"[runner] run_id={run_id} scenarios={len(scenarios)} sut={sut_base_url} "
         f"retrieval={retrieval_base_url} judge={judge_base_url} concurrency={concurrency}")

    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as sut_client, httpx.AsyncClient() as retrieval_client:
        async def _one(s: dict) -> ScenarioReport:
            async with sem:
                t0 = time.time()
                run = await _drive_one_scenario(
                    scenario=s, system_prompt=system_prompt,
                    sut_client=sut_client, sut_base_url=sut_base_url,
                    sut_api_key=sut_api_key,
                    retrieval_client=retrieval_client,
                    retrieval_base_url=retrieval_base_url,
                    retrieval_api_key=retrieval_api_key,
                    max_turns=max_turns, timeout_s=per_scenario_timeout_s,
                )
                # Enrich the scenario dict with runtime flags for the checks.
                s_runtime = dict(s)
                s_runtime["_recall_called"] = any(
                    (tc.get("function") or {}).get("name") == "recall_context"
                    for tc in run.tool_calls
                )
                s_runtime["_tool_calls"] = run.tool_calls
                programmatic = _checks.run_programmatic_checks(rubric, s_runtime, run.response_text)
                if use_judge:
                    judgement = await _judge.judge_response(
                        rubric=rubric, scenario=s_runtime,
                        response_text=run.response_text,
                        retrieved_hits=run.retrieved_hits,
                        tool_calls=run.tool_calls,
                        programmatic=programmatic,
                        base_url=judge_base_url, api_key=judge_api_key,
                        cache_dir=judge_cache_dir, use_cache=use_judge_cache,
                    )
                else:
                    # Promote programmatic-only scores to a judgement; mark
                    # the rest -1.
                    scores: dict[str, _judge.DimScore] = {}
                    for did in _judge.applicable_dimensions(s_runtime):
                        pr = programmatic.get(did)
                        if pr is not None and pr.score is not None:
                            scores[did] = _judge.DimScore(
                                score=pr.score,
                                reasoning="; ".join(pr.evidence) or "programmatic",
                                source="programmatic",
                            )
                        else:
                            scores[did] = _judge.DimScore(
                                score=-1, reasoning="judge disabled", source="judge",
                            )
                    judgement = _judge.JudgeResult(
                        scenario_id=s["id"],
                        rubric_version=int(rubric.get("version", 0)),
                        scores=scores,
                    )
                score, failed = _score_one(rubric, judgement)
                _log(f"[runner] {s['id']} score={score} failed_critical={failed} "
                     f"latency={run.latency_s:.1f}s elapsed={time.time()-t0:.1f}s")
                return ScenarioReport(
                    scenario=s, run=run, programmatic=programmatic,
                    judgement=judgement, score_0_100=score, failed_critical=failed,
                )

        reports: list[ScenarioReport] = await asyncio.gather(
            *(_one(s) for s in scenarios)
        )

    finished = time.time()
    summary = _summarize(rubric, reports)
    rr = RunReport(
        run_id=run_id, started_at=started, finished_at=finished,
        sut_base_url=sut_base_url, retrieval_base_url=retrieval_base_url,
        rubric_version=int(rubric.get("version", 0)),
        scenarios=reports, summary=summary,
    )
    _persist(rr, runs_dir)
    return rr


def _persist(report: RunReport, runs_dir: Path) -> None:
    out = runs_dir / report.run_id
    out.mkdir(parents=True, exist_ok=True)
    # results.jsonl — one line per scenario, full detail.
    with (out / "results.jsonl").open("w") as f:
        for r in report.scenarios:
            f.write(json.dumps({
                "scenario_id": r.scenario["id"],
                "intent": r.scenario.get("intent"),
                "register": r.scenario.get("register"),
                "score_0_100": r.score_0_100,
                "failed_critical": r.failed_critical,
                "sut_error": r.run.sut_error,
                "latency_s": round(r.run.latency_s, 2),
                "response_text": r.run.response_text,
                "tool_calls": r.run.tool_calls,
                "n_hits": len(r.run.retrieved_hits),
                "judgement": r.judgement.to_dict(),
                "programmatic": {
                    k: {"score": v.score, "evidence": v.evidence,
                        "programmatic_only": v.programmatic_only}
                    for k, v in r.programmatic.items()
                },
            }, ensure_ascii=False) + "\n")
    # summary.json — machine-readable
    (out / "summary.json").write_text(
        json.dumps({
            "run_id": report.run_id,
            "started_at": report.started_at,
            "finished_at": report.finished_at,
            "rubric_version": report.rubric_version,
            "sut_base_url": report.sut_base_url,
            "retrieval_base_url": report.retrieval_base_url,
            "summary": report.summary,
        }, indent=2, ensure_ascii=False),
    )
    # summary.md — human-readable
    (out / "summary.md").write_text(_render_summary_md(report))
    _log(f"[runner] wrote {out}/{{results.jsonl, summary.json, summary.md}}")


def _render_summary_md(report: RunReport) -> str:
    s = report.summary
    lines = [
        f"# Run {report.run_id}",
        "",
        f"- **Status**: {s.get('overall_status', '?')}",
        f"- **Overall**: {s.get('overall_score_0_100', 0)}/100",
        f"- **Scenarios**: {s.get('n_scenarios', 0)}",
        f"- **Rubric version**: {report.rubric_version}",
        f"- **Wall**: {round(report.finished_at - report.started_at, 1)}s",
        "",
        "## By intent",
        "| Intent | Score |",
        "|---|---|",
    ]
    for intent, mean in sorted(s.get("intent_means", {}).items()):
        lines.append(f"| {intent} | {mean} |")
    lines += [
        "",
        "## By dimension (mean across applicable scenarios; 0–2 scale)",
        "| Dimension | Mean |",
        "|---|---|",
    ]
    for did, mean in sorted(s.get("dim_means", {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"| {did} | {mean} |")
    crit = s.get("critical_failures_by_dim", {})
    if crit:
        lines += ["", "## CRITICAL failures (count of scenarios where dim scored 0)"]
        for did, n in sorted(crit.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{did}**: {n}")
    warns = s.get("high_dim_warnings", [])
    if warns:
        lines += ["", "## HIGH dimensions below warning threshold"]
        for did in warns:
            lines.append(f"- {did}")
    return "\n".join(lines) + "\n"
