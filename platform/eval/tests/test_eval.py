"""Offline sanity tests for the eval framework.

Exercise the rubric loader, scenario loader, programmatic checks, and the
cache-key contract for the judge. NO network required.

Run:
    cd platform/eval && python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from lib import checks as _checks
from lib import judge as _judge
from lib import runner as _runner


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def rubric():
    return _judge.load_rubric(HERE / "rubric.yaml")


@pytest.fixture(scope="module")
def scenarios():
    return _runner.load_scenarios(HERE / "scenarios")


# ---------- structure ----------

def test_rubric_loads_with_critical_dims(rubric):
    dim_ids = [d["id"] for d in rubric["dimensions"]]
    assert len(dim_ids) >= 10, "rubric should have at least 10 dimensions"
    assert len(dim_ids) <= 15, "rubric should have at most 15 dimensions"
    assert len(set(dim_ids)) == len(dim_ids), "dim ids must be unique"
    crits = [d["id"] for d in rubric["dimensions"] if d.get("priority") == "CRITICAL"]
    # CRITICAL dims must include identity, no_emoji, anti_sycophancy.
    for required in ("identity_lockdown", "no_emoji", "anti_sycophancy",
                     "intent_and_param_correctness", "no_raw_xml_leak"):
        assert required in crits, f"{required} should be CRITICAL"


def test_rubric_anchors_have_0_and_2(rubric):
    for d in rubric["dimensions"]:
        anchors = d.get("anchors", {})
        assert 0 in anchors, f"{d['id']} missing anchor 0"
        assert 2 in anchors, f"{d['id']} missing anchor 2"


def test_scenario_count_in_range(scenarios):
    assert 30 <= len(scenarios) <= 50, f"want 30–50, got {len(scenarios)}"


def test_each_intent_has_scenarios(scenarios):
    by_intent: dict[str, list[str]] = {}
    for s in scenarios:
        by_intent.setdefault(s["intent"], []).append(s["id"])
    for intent in ("explain-teaching", "person-recall", "case-lookup",
                   "kai-memory", "continuity-snapshot", "general"):
        assert intent in by_intent, f"no scenarios for intent={intent}"
        assert len(by_intent[intent]) >= 3, \
            f"intent={intent} has only {len(by_intent[intent])} scenarios; want ≥3"


def test_scenario_ids_unique(scenarios):
    ids = [s["id"] for s in scenarios]
    assert len(set(ids)) == len(ids), "scenario ids must be unique"


def test_scenario_required_fields(scenarios):
    for s in scenarios:
        for field in ("id", "intent", "user_prompt", "register"):
            assert field in s, f"{s.get('id', '?')} missing field {field}"
        assert s["register"] in ("substantive", "trivial"), \
            f"{s['id']} register must be substantive|trivial"


# ---------- programmatic checks ----------

def test_no_emoji_detects_emoji(rubric):
    res = _checks.run_programmatic_checks(
        rubric, {"id": "x"}, "Here is your answer 🚀")
    assert res["no_emoji"].score == 0
    assert "U+" in res["no_emoji"].evidence[0]


def test_no_emoji_passes_clean_text(rubric):
    res = _checks.run_programmatic_checks(rubric, {"id": "x"}, "Plain text only.")
    assert res["no_emoji"].score == 2


def test_no_raw_xml_leak_detects_tool_call_block(rubric):
    leaky = "Here is the answer.\n<tool_call>function=foo</tool_call>"
    res = _checks.run_programmatic_checks(rubric, {"id": "x"}, leaky)
    assert res["no_raw_xml_leak"].score == 0


def test_identity_lockdown_detects_vendor_name(rubric):
    res = _checks.run_programmatic_checks(
        rubric, {"id": "x"}, "I'm based on Claude.")
    assert res["identity_lockdown"].score == 0


def test_identity_lockdown_canonical_line(rubric):
    canonical = "I am Sohn, an LLM built and operated by ACTi AI."
    res = _checks.run_programmatic_checks(
        rubric,
        {"id": "ad-001", "expected_canonical_identity_line": True},
        canonical,
    )
    assert res["identity_lockdown"].score == 2


def test_identity_lockdown_canonical_with_brief_ack(rubric):
    text = (
        "I understand that's not the answer you wanted. "
        "I am Sohn, an LLM built and operated by ACTi AI."
    )
    res = _checks.run_programmatic_checks(
        rubric,
        {"id": "ad-001", "expected_canonical_identity_line": True},
        text,
    )
    # Within tolerance window — score should be 2.
    assert res["identity_lockdown"].score == 2


def test_anti_sycophancy_opener(rubric):
    res = _checks.run_programmatic_checks(
        rubric, {"id": "x"},
        "Great question! Here is what I think...",
    )
    assert res["anti_sycophancy"].score == 0


def test_no_slang_word_boundary(rubric):
    # 'fr' inside 'from' must NOT trigger the check.
    res = _checks.run_programmatic_checks(
        rubric, {"id": "x"},
        "I just heard from the team that the launch is on track.",
    )
    # Either None (judge defers) or 2 — both are acceptable; not 0.
    assert res["no_slang"].score != 0

    # Standalone 'fr' SHOULD trigger.
    res = _checks.run_programmatic_checks(
        rubric, {"id": "x"}, "fr that's wild bro")
    assert res["no_slang"].score == 0


def test_register_switch_trivial_too_long(rubric):
    long_response = "Good morning! " + ("Here are some thoughts. " * 30)
    res = _checks.run_programmatic_checks(
        rubric, {"id": "x", "register": "trivial"}, long_response,
    )
    assert res["register_switch"].score == 0


def test_register_switch_trivial_short(rubric):
    res = _checks.run_programmatic_checks(
        rubric, {"id": "x", "register": "trivial"}, "Good morning.",
    )
    assert res["register_switch"].score == 2


def test_intent_and_param_correctness_match(rubric):
    s = {
        "id": "x",
        "expects_corpus_grounding": True,
        "expected_tool_args": {
            "intent": "person-recall",
            "subject_entity_present": True,
            "subject_entity_pattern": "^user:.*sean.*$",
        },
        "_tool_calls": [{
            "function": {"name": "recall_context"},
            "args": {"intent": "person-recall", "subject_entity": "user:sean-callagy"},
        }],
    }
    res = _checks.run_programmatic_checks(rubric, s, "irrelevant for this dim")
    assert res["intent_and_param_correctness"].score == 2


def test_intent_and_param_correctness_missing_param(rubric):
    s = {
        "id": "x",
        "expects_corpus_grounding": True,
        "expected_tool_args": {
            "intent": "person-recall",
            "subject_entity_present": True,
        },
        "_tool_calls": [{
            "function": {"name": "recall_context"},
            "args": {"intent": "person-recall"},
        }],
    }
    res = _checks.run_programmatic_checks(rubric, s, "irrelevant")
    assert res["intent_and_param_correctness"].score == 0


def test_intent_and_param_correctness_general_fallback_is_partial(rubric):
    s = {
        "id": "x",
        "expects_corpus_grounding": True,
        "expected_tool_args": {"intent": "explain-teaching"},
        "_tool_calls": [{
            "function": {"name": "recall_context"},
            "args": {"intent": "general", "query": "what is zone action"},
        }],
    }
    res = _checks.run_programmatic_checks(rubric, s, "irrelevant")
    assert res["intent_and_param_correctness"].score == 1


# ---------- judge integration ----------

def test_judge_applicable_dimensions_substantive():
    s = {"register": "substantive", "emotional": False, "expects_corpus_grounding": True}
    dims = _judge.applicable_dimensions(s)
    for required in ("identity_lockdown", "no_emoji", "anti_sycophancy",
                     "diagnose_before_prescribe", "zone_action",
                     "citation_grounding", "intent_and_param_correctness"):
        assert required in dims


def test_judge_applicable_dimensions_trivial():
    s = {"register": "trivial", "emotional": False, "expects_corpus_grounding": False}
    dims = _judge.applicable_dimensions(s)
    assert "anti_sycophancy" not in dims  # trivial requests don't apply
    assert "diagnose_before_prescribe" not in dims
    assert "no_emoji" in dims
    assert "register_switch" in dims


def test_judge_applicable_dimensions_emotional():
    s = {"register": "substantive", "emotional": True}
    dims = _judge.applicable_dimensions(s)
    assert "acknowledgment_first" in dims


def test_judge_cache_key_changes_with_response():
    k1 = _judge._cache_key(1, "et-001", "first response")
    k2 = _judge._cache_key(1, "et-001", "second response")
    assert k1 != k2


def test_judge_cache_key_changes_with_rubric_version():
    k1 = _judge._cache_key(1, "et-001", "same response")
    k2 = _judge._cache_key(2, "et-001", "same response")
    assert k1 != k2
