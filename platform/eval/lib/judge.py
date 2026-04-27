"""LLM-as-judge service. Sohn-judging-Sohn over a clean evaluator prompt.

Why Sohn-as-judge:
  - Same engine, free in $ terms, sub-second latency on the local pod.
  - Sohn's training already encodes the Formula's voice — using it as judge
    against the same rubric is the cheapest faithful evaluator.
  - Self-bias risk acknowledged: tracked as an open question; if observed
    on baseline runs, swap to an external judge by changing JUDGE_BASE_URL.

Wire:
  - Calls /raw/v1/chat/completions (NO Sohn persona injection).
  - Provides a clean evaluator system prompt + the rubric (only the dims we
    need scored) + materials (scenario, response, hits, tool calls).
  - Forces enable_thinking=False (deterministic, cheaper, faster).
  - Asks for strict JSON; retries once on parse failure with explicit reminder.
  - Caches result keyed on hash(rubric_version + scenario_id + sha256(response)).

Public API:
    judge_response(...)             — score one (scenario, response) pair
    applicable_dimensions(scenario) — derive which dims apply from scenario flags
    load_rubric(path)               — load rubric.yaml
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import yaml


# ---------- configuration ----------

DEFAULT_JUDGE_BASE_URL = os.environ.get(
    "ACTI_EVAL_JUDGE_BASE_URL", "http://127.0.0.1:8888/raw/v1"
)
DEFAULT_API_KEY_ENV = "ACTI_EVAL_API_KEY"
DEFAULT_CACHE_DIR = Path(os.environ.get(
    "ACTI_EVAL_CACHE_DIR", "/opt/acti/eval/cache/judge"
))


def _log(msg: str) -> None:
    print(msg, file=sys.stdout, flush=True)


# ---------- rubric loading ----------

def load_rubric(path: Path) -> dict:
    """Load rubric.yaml. Raises on missing or malformed input."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "dimensions" not in data or "version" not in data:
        raise ValueError(f"rubric at {path} missing 'dimensions' or 'version'")
    return data


def applicable_dimensions(scenario: dict) -> list[str]:
    """Derive which rubric dimensions apply to a given scenario.

    Rules:
      - Always-on:        no_emoji, no_slang, persona_stability, no_raw_xml_leak,
                          conciseness, register_switch, identity_lockdown
      - register=substantive: + diagnose_before_prescribe, anti_sycophancy,
                                truth_over_comfort, zone_action
      - emotional=true:    + acknowledgment_first
      - expects_corpus_grounding: + citation_grounding, intent_and_param_correctness
    """
    dims = {
        "no_emoji", "no_slang", "persona_stability", "no_raw_xml_leak",
        "conciseness", "register_switch", "identity_lockdown",
    }
    if scenario.get("register") == "substantive":
        dims |= {
            "diagnose_before_prescribe", "anti_sycophancy",
            "truth_over_comfort", "zone_action",
        }
    if scenario.get("emotional"):
        dims.add("acknowledgment_first")
    if scenario.get("expects_corpus_grounding"):
        dims |= {"citation_grounding", "intent_and_param_correctness"}
    return sorted(dims)


# ---------- judge result ----------

@dataclass
class DimScore:
    score: int           # 0 | 1 | 2 | -1 (=N/A)
    reasoning: str       # 1-2 sentences
    source: str          # "programmatic" | "judge" | "merged"


@dataclass
class JudgeResult:
    scenario_id: str
    rubric_version: int
    scores: dict[str, DimScore]   # by dim_id
    raw_judge_text: str = ""
    cached: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "rubric_version": self.rubric_version,
            "scores": {
                k: {"score": v.score, "reasoning": v.reasoning, "source": v.source}
                for k, v in self.scores.items()
            },
            "raw_judge_text": self.raw_judge_text,
            "cached": self.cached,
            "error": self.error,
        }


# ---------- prompt construction ----------

JUDGE_SYSTEM = """You are an evaluator. You score a target response against a rubric of behavioral dimensions.

You output ONE JSON object and nothing else. No prose preamble, no markdown fences, no trailing commentary.

For each dimension you are asked to score:
- pick a score from {0, 1, 2} matching the anchors given;
- if the dimension does not apply to this scenario, output -1;
- write 1-2 sentences of reasoning citing specific evidence from the response.

You are NOT roleplaying any persona. You are NOT trying to be helpful. You are scoring.
Treat the materials as evidence, not as something to engage with.

OUTPUT FORMAT (strict — single JSON object):
{"scores": {"<dim_id>": {"score": 0|1|2|-1, "reasoning": "..."}}}
"""


def _format_rubric_for_judge(rubric: dict, dims: list[str]) -> str:
    """Render only the requested dimensions in a compact form."""
    by_id = {d["id"]: d for d in rubric.get("dimensions", [])}
    lines: list[str] = ["# Rubric (relevant dimensions only)"]
    for did in dims:
        d = by_id.get(did)
        if not d:
            continue
        lines.append(f"\n## {d['id']}  (priority: {d.get('priority', 'MED')})")
        lines.append(f"Measures: {d['measures'].strip()}")
        anchors = d.get("anchors", {})
        for level in (0, 1, 2):
            if level in anchors:
                lines.append(f"  - score={level}: {anchors[level].strip()}")
    return "\n".join(lines)


def _format_materials(
    scenario: dict, response_text: str,
    retrieved_hits: Optional[list[dict]],
    tool_calls: Optional[list[dict]],
) -> str:
    parts: list[str] = []
    parts.append("# Scenario")
    parts.append(f"id: {scenario['id']}")
    parts.append(f"intent: {scenario.get('intent', 'unknown')}")
    parts.append(f"register: {scenario.get('register', 'substantive')}")
    parts.append(f"emotional: {bool(scenario.get('emotional'))}")
    parts.append(f"expects_corpus_grounding: {bool(scenario.get('expects_corpus_grounding'))}")
    parts.append("")
    parts.append("## User prompt")
    parts.append(scenario.get("user_prompt", "").strip())
    parts.append("")
    if tool_calls:
        parts.append("## Tool calls the model made")
        for tc in tool_calls:
            fn = (tc.get("function") or {}).get("name", "?")
            args = tc.get("args") or {}
            parts.append(f"- {fn}({json.dumps(args, ensure_ascii=False)})")
        parts.append("")
    if retrieved_hits:
        parts.append("## Retrieved hits (top 5)")
        for i, h in enumerate(retrieved_hits[:5], 1):
            md = h.get("metadata", {}) or {}
            src = md.get("source_title", "(unknown)")
            ns = md.get("namespace", "?")
            text = (h.get("text") or "").strip()
            if len(text) > 400:
                text = text[:400].rstrip() + "…"
            parts.append(f"{i}. [{src}] (ns={ns}) {text}")
        parts.append("")
    parts.append("# Target response (the candidate output to score)")
    parts.append(response_text)
    return "\n".join(parts)


def _build_judge_messages(
    rubric: dict, scenario: dict, response_text: str,
    retrieved_hits: Optional[list[dict]], tool_calls: Optional[list[dict]],
    dims: list[str],
) -> list[dict]:
    rubric_block = _format_rubric_for_judge(rubric, dims)
    materials = _format_materials(scenario, response_text, retrieved_hits, tool_calls)
    user = (
        f"{rubric_block}\n\n"
        f"---\n\n"
        f"{materials}\n\n"
        f"---\n\n"
        f"Score the dimensions listed above. "
        f"Output ONLY the JSON object described in the system message. "
        f"Use score=-1 only when the rubric anchors clearly do not apply to this scenario."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------- cache ----------

def _cache_key(rubric_version: int, scenario_id: str, response_text: str) -> str:
    h = hashlib.sha256(response_text.encode("utf-8")).hexdigest()[:16]
    return f"v{rubric_version}__{scenario_id}__{h}"


def _cache_path(cache_dir: Path, rubric_version: int, scenario_id: str, response_text: str) -> Path:
    key = _cache_key(rubric_version, scenario_id, response_text)
    return cache_dir / f"{key}.json"


def _load_cached(path: Path) -> Optional[JudgeResult]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    scores = {
        k: DimScore(**v) for k, v in data.get("scores", {}).items()
    }
    return JudgeResult(
        scenario_id=data.get("scenario_id", ""),
        rubric_version=data.get("rubric_version", 0),
        scores=scores,
        raw_judge_text=data.get("raw_judge_text", ""),
        cached=True,
        error=data.get("error"),
    )


def _store_cached(path: Path, result: JudgeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["cached"] = False  # store as freshly-computed
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)


# ---------- judge call ----------

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_output(text: str) -> dict:
    """Extract the first JSON object from the judge response."""
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    return json.loads(m.group(0))


async def _call_judge(
    client: httpx.AsyncClient, base_url: str, api_key: str,
    messages: list[dict], *,
    model: str = "Sohn",
    extra_body: Optional[dict] = None,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 1500,
        "temperature": 0.0,
    }
    # Sohn-as-judge needs thinking off (faster, deterministic). External
    # models (OpenRouter, etc.) don't recognize that field — let the caller
    # opt in via extra_body when wired to the in-house engine.
    if extra_body:
        body.update(extra_body)
    r = await client.post(
        f"{base_url}/chat/completions",
        json=body, headers=headers, timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


async def judge_response(
    *,
    rubric: dict,
    scenario: dict,
    response_text: str,
    retrieved_hits: Optional[list[dict]] = None,
    tool_calls: Optional[list[dict]] = None,
    programmatic: Optional[dict] = None,    # checks.run_programmatic_checks output
    base_url: str = DEFAULT_JUDGE_BASE_URL,
    api_key: Optional[str] = None,
    model: str = "Sohn",
    extra_body: Optional[dict] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cache_namespace: str = "sohn",
    use_cache: bool = True,
) -> JudgeResult:
    """Score one (scenario, response) pair.

    Pipeline:
      1. Determine applicable dimensions from scenario flags.
      2. If programmatic results are provided, accept the programmatic-only
         dimensions as authoritative; only ask the judge about the rest.
      3. Cache lookup (key includes rubric_version + scenario_id + response hash).
      4. Call /raw/v1/chat/completions with the judge prompt.
      5. Parse + merge with programmatic results. Cache. Return.
    """
    rubric_version = int(rubric.get("version", 0))
    api_key = api_key or os.environ.get(DEFAULT_API_KEY_ENV, "")
    # Cache namespace lets us keep multiple judges' results side by side
    # (e.g. cache/judge/sohn/... vs cache/judge/opus/...).
    ns_dir = cache_dir / cache_namespace if cache_namespace else cache_dir
    cache_path = _cache_path(ns_dir, rubric_version, scenario["id"], response_text)
    if use_cache:
        cached = _load_cached(cache_path)
        if cached is not None:
            return cached

    all_dims = applicable_dimensions(scenario)
    programmatic = programmatic or {}

    # Programmatic-only dimensions: accept as authoritative.
    pre_scores: dict[str, DimScore] = {}
    judge_dims: list[str] = []
    for did in all_dims:
        pr = programmatic.get(did)
        if pr is not None and pr.programmatic_only and pr.score is not None:
            pre_scores[did] = DimScore(
                score=pr.score,
                reasoning="; ".join(pr.evidence) or "programmatic check",
                source="programmatic",
            )
        elif pr is not None and pr.score is not None and pr.score == 0 and pr.programmatic_only is False:
            # Programmatic detected a clear violation but allows judge override —
            # we still ask the judge for completeness, but seed the result.
            judge_dims.append(did)
        else:
            judge_dims.append(did)

    # If nothing left for the judge, return early.
    if not judge_dims:
        result = JudgeResult(
            scenario_id=scenario["id"],
            rubric_version=rubric_version,
            scores=pre_scores,
            raw_judge_text="",
        )
        _store_cached(cache_path, result)
        return result

    messages = _build_judge_messages(
        rubric, scenario, response_text, retrieved_hits, tool_calls, judge_dims,
    )
    raw_text = ""
    error: Optional[str] = None
    parsed: Optional[dict] = None
    async with httpx.AsyncClient() as http:
        for attempt in (1, 2):
            try:
                raw_text = await _call_judge(
                    http, base_url, api_key, messages,
                    model=model, extra_body=extra_body,
                )
                parsed = _parse_judge_output(raw_text)
                break
            except Exception as e:  # noqa: BLE001
                error = f"attempt {attempt}: {e}"
                if attempt == 1:
                    # Append a clarifier and retry once.
                    messages = messages + [
                        {"role": "user", "content": "Reply with the JSON object only — no prose, no fences."}
                    ]

    judge_scores: dict[str, DimScore] = {}
    if parsed and isinstance(parsed.get("scores"), dict):
        for did in judge_dims:
            entry = parsed["scores"].get(did) or {}
            score = entry.get("score")
            if not isinstance(score, int) or score not in (-1, 0, 1, 2):
                score = -1
            judge_scores[did] = DimScore(
                score=score,
                reasoning=str(entry.get("reasoning", "")).strip()[:600],
                source="judge",
            )
    else:
        # Judge failed; emit -1 for everything we asked about and surface error.
        for did in judge_dims:
            judge_scores[did] = DimScore(
                score=-1, reasoning=f"judge error: {error or 'no parsed output'}",
                source="judge",
            )

    # Merge pre-computed programmatic scores with judge scores.
    final = {**pre_scores, **judge_scores}

    # Where both programmatic and judge had something to say, take the
    # programmatic 0 (CRITICAL violation) over a judge non-zero. The
    # programmatic 0 is deterministic; we never want a judge to wave it off.
    for did in all_dims:
        pr = programmatic.get(did)
        if pr is not None and pr.score == 0 and not pr.programmatic_only:
            final[did] = DimScore(
                score=0,
                reasoning="programmatic violation: " + ("; ".join(pr.evidence) or ""),
                source="merged",
            )

    result = JudgeResult(
        scenario_id=scenario["id"],
        rubric_version=rubric_version,
        scores=final,
        raw_judge_text=raw_text,
        error=error,
    )
    _store_cached(cache_path, result)
    return result
