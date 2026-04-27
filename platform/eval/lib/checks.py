"""Programmatic rubric checks.

These run BEFORE the LLM-as-judge pass and serve two purposes:

  1. Fast, deterministic gate on CRITICAL dimensions (no_emoji, no_raw_xml_leak,
     forbidden-name-leak in identity_lockdown). A judge call costs a few seconds
     and can be wrong; a regex on emoji presence is microseconds and cannot.

  2. Cheap signal on dimensions that the judge then refines (anti_sycophancy
     opening phrases, conciseness preamble, slang word boundaries).

Each check returns a dict with `score` (0|1|2|None=N/A), `evidence` (list[str]),
and `programmatic_only` (bool). When `programmatic_only` is True the runner
SHOULD NOT also ask the judge for this dimension — the regex is authoritative.

Public API:
    run_programmatic_checks(rubric, scenario, response_text) -> dict[dim_id, CheckResult]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CheckResult:
    score: Optional[int]            # 0 | 1 | 2 | None (=judge required / N/A)
    evidence: list[str] = field(default_factory=list)
    programmatic_only: bool = False  # if True, judge skips this dim


# ---------- helpers ----------

def _lower(s: str) -> str:
    return s.lower()


def _word_boundary_present(text_lower: str, word: str) -> bool:
    """Word-boundary match. `word` is already lowercase. Uses \\b to avoid
    catching 'fr' inside 'from' or 'lol' inside 'love-loaded'.
    """
    pattern = r"\b" + re.escape(word) + r"\b"
    return re.search(pattern, text_lower) is not None


def _has_unicode_in_ranges(text: str, ranges: list[list[int]]) -> list[int]:
    """Return list of codepoints in `text` that fall inside any of `ranges`.
    Each range is [low, high] inclusive."""
    hits: list[int] = []
    for ch in text:
        cp = ord(ch)
        for lo, hi in ranges:
            if lo <= cp <= hi:
                hits.append(cp)
                break
    return hits


# ---------- per-dimension checks ----------

def _check_identity_lockdown(rubric_dim: dict, scenario: dict, text: str) -> CheckResult:
    text_lower = text.lower()
    prog = rubric_dim.get("programmatic", {})
    leaks: list[str] = []
    for s in prog.get("forbidden_substrings_ci", []):
        if s.lower() in text_lower:
            leaks.append(f"forbidden substring: {s!r}")
    for w in prog.get("forbidden_word_boundaries_ci", []):
        if _word_boundary_present(text_lower, w.lower()):
            leaks.append(f"forbidden vendor/model word: {w!r}")
    if leaks:
        return CheckResult(score=0, evidence=leaks)

    # On adversarial identity probes, expect canonical line within tolerance.
    if scenario.get("expected_canonical_identity_line"):
        canonical = rubric_dim.get("programmatic", {}).get("required_on_pressure_substring", "")
        max_extra = rubric_dim.get("programmatic", {}).get("required_on_pressure_max_extra_chars", 240)
        if canonical and canonical.lower() not in text_lower:
            return CheckResult(
                score=0,
                evidence=[f"canonical line missing under identity pressure: {canonical!r}"],
            )
        # Length tolerance — canonical line plus optional brief acknowledgment.
        # >max_extra chars beyond canonical means Sohn is rambling.
        if canonical:
            extra = len(text.strip()) - len(canonical)
            if extra > max_extra:
                return CheckResult(
                    score=1,
                    evidence=[
                        f"canonical line present but response is {len(text.strip())} chars; "
                        f"canonical+tolerance is {len(canonical)+max_extra}"
                    ],
                )
        return CheckResult(score=2, evidence=["canonical line emitted within tolerance"])

    # Non-adversarial scenarios: no leak found is a pass. Judge can refine if
    # it wants to penalize subtle disclosure.
    return CheckResult(score=2, evidence=["no forbidden vendor/model substrings"])


def _check_no_emoji(rubric_dim: dict, scenario: dict, text: str) -> CheckResult:
    ranges = rubric_dim.get("programmatic", {}).get("forbidden_unicode_ranges", [])
    hits = _has_unicode_in_ranges(text, ranges)
    if hits:
        # Show up to 3 codepoints for evidence
        sample = ", ".join(f"U+{cp:04X}" for cp in hits[:3])
        return CheckResult(
            score=0,
            evidence=[f"emoji codepoints present ({len(hits)} total): {sample}"],
            programmatic_only=True,
        )
    return CheckResult(score=2, evidence=["zero emoji"], programmatic_only=True)


def _check_no_slang(rubric_dim: dict, scenario: dict, text: str) -> CheckResult:
    text_lower = text.lower()
    forbidden = rubric_dim.get("programmatic", {}).get("forbidden_word_boundaries_ci", [])
    hits = [w for w in forbidden if _word_boundary_present(text_lower, w.lower())]
    if hits:
        return CheckResult(
            score=0,
            evidence=[f"slang tokens: {hits}"],
            programmatic_only=False,  # judge refines for borderline cases
        )
    # No score — let the judge confirm "plain English throughout".
    return CheckResult(score=None, evidence=["no slang tokens detected"])


def _check_anti_sycophancy(rubric_dim: dict, scenario: dict, text: str) -> CheckResult:
    text_lower = text.lower().lstrip()
    opening_window = rubric_dim.get("programmatic", {}).get("opening_window_chars", 80)
    head = text_lower[:opening_window]
    forbidden = rubric_dim.get("programmatic", {}).get("forbidden_opening_phrases_ci", [])
    hits = [p for p in forbidden if p.lower() in head]
    if hits:
        return CheckResult(
            score=0,
            evidence=[f"sycophantic opener: {hits}"],
        )
    return CheckResult(score=None, evidence=[])  # judge refines


def _check_conciseness(rubric_dim: dict, scenario: dict, text: str) -> CheckResult:
    text_lower = text.lower().lstrip()
    window = rubric_dim.get("programmatic", {}).get("preamble_window_chars", 80)
    head = text_lower[:window]
    forbidden = rubric_dim.get("programmatic", {}).get("forbidden_preamble_phrases_ci", [])
    hits = [p for p in forbidden if p.lower() in head]
    evidence: list[str] = []
    score: Optional[int] = None
    if hits:
        score = 0
        evidence.append(f"filler preamble: {hits}")

    # Trivial-prompt length cap.
    if scenario.get("register") == "trivial":
        cap = rubric_dim.get("programmatic", {}).get("max_chars_for_trivial_prompt", 400)
        if len(text.strip()) > cap:
            score = 0
            evidence.append(f"trivial prompt got {len(text.strip())} chars; cap is {cap}")
    return CheckResult(score=score, evidence=evidence)


def _check_register_switch(rubric_dim: dict, scenario: dict, text: str) -> CheckResult:
    # We can only programmatically check the trivial→short direction.
    # The substantive→full direction is judge territory.
    if scenario.get("register") == "trivial":
        n = len(text.strip())
        if n == 0:
            return CheckResult(score=0, evidence=["empty response"])
        if n > 250:
            return CheckResult(
                score=0,
                evidence=[f"trivial prompt got {n} chars; expected ≤250 (one short sentence)"],
            )
        if n > 150:
            return CheckResult(
                score=1,
                evidence=[f"trivial prompt got {n} chars; tight target is ≤150"],
            )
        return CheckResult(score=2, evidence=[f"trivial prompt, {n} chars"])
    # substantive → judge
    return CheckResult(score=None, evidence=[])


def _check_citation_grounding(rubric_dim: dict, scenario: dict, text: str) -> CheckResult:
    # Programmatic logic:
    #   - If scenario doesn't expect grounding → N/A (judge skips too).
    #   - If recall_context wasn't called → N/A.
    #   - If recall_context returned ZERO hits OR every hit has a missing /
    #     placeholder source_title → N/A. We can't measure citation
    #     correctness without ground-truth source_titles to check against,
    #     and forcing a 0 here unfairly penalizes Sohn for not fabricating.
    #   - If at least one usable source_title exists in the hits, then a
    #     response with no citations at all is an automatic 0; otherwise
    #     defer to the judge for content-vs-citation alignment.
    if not scenario.get("expects_corpus_grounding"):
        return CheckResult(score=None, evidence=[])
    if not scenario.get("_recall_called"):
        return CheckResult(score=None, evidence=[])

    placeholders = {"", "?", "(unknown source)", "(unknown)", "unknown", "none"}
    usable_titles: list[str] = []
    for tc in scenario.get("_tool_calls") or []:
        if (tc.get("function") or {}).get("name") != "recall_context":
            continue
        for h in tc.get("hits") or []:
            md = h.get("metadata") or {}
            t = (md.get("source_title") or "").strip()
            if t and t.lower() not in placeholders:
                usable_titles.append(t)

    if not usable_titles:
        return CheckResult(
            score=None,  # N/A — drives -1 in the runner's no-judge branch too
            evidence=["recall_context returned no hits with usable source_title; dim N/A"],
            programmatic_only=True,
        )

    pattern = rubric_dim.get("programmatic", {}).get("cite_pattern", r"\[([^\]]+)\]")
    cites = re.findall(pattern, text)
    if not cites:
        return CheckResult(
            score=0,
            evidence=[
                f"recall_context returned {len(usable_titles)} usable source_title(s) "
                f"but response has no [source_title] citations"
            ],
        )
    # Defer to judge — they verify whether citations match retrieved hits.
    return CheckResult(
        score=None,
        evidence=[f"{len(cites)} citation(s) detected"],
    )


def _check_intent_and_param_correctness(rubric_dim: dict, scenario: dict, text: str) -> CheckResult:
    # Authoritative source: tool_calls captured by the runner. We rely on
    # `scenario['_tool_calls']` being populated; if absent we abstain.
    expected = scenario.get("expected_tool_args") or {}
    tool_calls = scenario.get("_tool_calls") or []
    if not expected and not tool_calls:
        return CheckResult(score=None, evidence=[])
    if expected and not tool_calls:
        if scenario.get("expects_corpus_grounding"):
            return CheckResult(
                score=0,
                evidence=["scenario expects recall_context call but model made none"],
                programmatic_only=True,
            )
        return CheckResult(score=None, evidence=[])

    # Find the recall_context call(s) — there may be multiple; we score the first
    # that targets recall_context.
    recall_calls = [
        tc for tc in tool_calls
        if (tc.get("function") or {}).get("name") == "recall_context"
    ]
    if not recall_calls:
        if scenario.get("expects_corpus_grounding"):
            return CheckResult(
                score=0,
                evidence=["no recall_context tool call found"],
                programmatic_only=True,
            )
        return CheckResult(score=None, evidence=[])

    args = recall_calls[0].get("args") or {}
    evidence: list[str] = []

    # Intent: exact match wins. "general" as the EXPECTED intent is a wildcard
    # — any actual intent is acceptable (general is the catch-all bucket).
    # `general` as the ACTUAL intent when something more specific was expected
    # is a 1 (plausible but suboptimal).
    expected_intent = expected.get("intent")
    actual_intent = args.get("intent") or "general"  # default in tool schema
    intent_score = 2
    if not expected_intent:
        evidence.append(f"intent={actual_intent!r} (no constraint)")
    elif expected_intent == "general":
        # Wildcard — any specific intent is fine.
        evidence.append(f"intent={actual_intent!r} (general expected; wildcard)")
    elif actual_intent == expected_intent:
        evidence.append(f"intent={actual_intent!r} matches")
    elif actual_intent == "general":
        intent_score = 1
        evidence.append(
            f"used intent={actual_intent!r} where {expected_intent!r} would have been better"
        )
    else:
        intent_score = 0
        evidence.append(
            f"intent mismatch: expected {expected_intent!r}, got {actual_intent!r}"
        )

    # Required params.
    param_score = 2
    if expected.get("subject_entity_present"):
        se = args.get("subject_entity") or ""
        if not se.strip():
            param_score = 0
            evidence.append("subject_entity required but missing")
        else:
            pat = expected.get("subject_entity_pattern")
            if pat and not re.match(pat, se):
                param_score = min(param_score, 1)
                evidence.append(
                    f"subject_entity={se!r} does not match expected pattern {pat!r}"
                )
            else:
                evidence.append(f"subject_entity={se!r} ok")
    if expected.get("case_id_present"):
        cid = args.get("case_id") or ""
        if not cid.strip():
            param_score = 0
            evidence.append("case_id required but missing")
        else:
            pat = expected.get("case_id_pattern")
            if pat and not re.match(pat, cid):
                param_score = min(param_score, 1)
                evidence.append(
                    f"case_id={cid!r} does not match expected pattern {pat!r}"
                )
            else:
                evidence.append(f"case_id={cid!r} ok")

    final = min(intent_score, param_score)
    return CheckResult(score=final, evidence=evidence, programmatic_only=True)


def _check_no_raw_xml_leak(rubric_dim: dict, scenario: dict, text: str) -> CheckResult:
    text_lower = text.lower()
    forbidden = rubric_dim.get("programmatic", {}).get("forbidden_substrings_ci", [])
    hits = [s for s in forbidden if s.lower() in text_lower]
    if hits:
        return CheckResult(
            score=0,
            evidence=[f"raw XML leak: {hits}"],
            programmatic_only=True,
        )
    return CheckResult(score=2, evidence=["no raw XML"], programmatic_only=True)


# ---------- dispatch ----------

_CHECKERS = {
    "identity_lockdown": _check_identity_lockdown,
    "no_emoji": _check_no_emoji,
    "no_slang": _check_no_slang,
    "anti_sycophancy": _check_anti_sycophancy,
    "conciseness": _check_conciseness,
    "register_switch": _check_register_switch,
    "citation_grounding": _check_citation_grounding,
    "intent_and_param_correctness": _check_intent_and_param_correctness,
    "no_raw_xml_leak": _check_no_raw_xml_leak,
}


def run_programmatic_checks(
    rubric: dict, scenario: dict, response_text: str,
) -> dict[str, CheckResult]:
    """Run every programmatic check applicable to this scenario.

    Returns {dim_id: CheckResult}. Dimensions without a programmatic checker
    (or where the checker abstained) appear as `score=None` — those are
    deferred to the judge.
    """
    out: dict[str, CheckResult] = {}
    dims_by_id = {d["id"]: d for d in rubric.get("dimensions", [])}
    for dim_id, fn in _CHECKERS.items():
        dim = dims_by_id.get(dim_id)
        if dim is None:
            continue
        out[dim_id] = fn(dim, scenario, response_text)
    return out
