"""Tests for lib/corpus_vocab.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.corpus_vocab import (  # noqa: E402
    CONTENT_TYPES,
    FIXED_SUBJECT_ENTITIES,
    FORBIDDEN_VENDOR_TOKENS,
    INTENTS,
    INTENT_REQUIRED_PARAMS,
    KNOWN_USER_SLUGS,
    NAMESPACES,
    NAMESPACE_SIZES,
    TOPICS,
    is_valid_subject_entity,
    normalize_user_slug,
)


# ---------- frozen-set sanity ----------


def test_namespaces_frozen() -> None:
    assert isinstance(NAMESPACES, frozenset)
    assert "teachings" in NAMESPACES
    assert len(NAMESPACES) == 6


def test_namespace_sizes_match_namespaces() -> None:
    """Every namespace has a size; sizes are positive."""
    assert set(NAMESPACE_SIZES.keys()) == set(NAMESPACES)
    for ns, n in NAMESPACE_SIZES.items():
        assert n > 0, f"{ns} has non-positive size"


def test_namespace_sizes_match_real_corpus() -> None:
    """The 2026-04-27 canonical-store scan numbers."""
    assert NAMESPACE_SIZES["teachings"] == 157_175
    assert NAMESPACE_SIZES["users"] == 25_052
    assert NAMESPACE_SIZES["memory"] == 3_829
    assert NAMESPACE_SIZES["products"] == 786
    assert NAMESPACE_SIZES["cases"] == 149
    assert NAMESPACE_SIZES["identity"] == 35


def test_content_types_includes_all_chunk_id_prefixes() -> None:
    """The chunk_id grammar's content_type slot uses these tokens."""
    assert "teaching" in CONTENT_TYPES
    assert "user-note" in CONTENT_TYPES
    assert "case-folder" in CONTENT_TYPES
    assert "memory-daily" in CONTENT_TYPES
    assert "memory-longterm" in CONTENT_TYPES
    assert "identity" in CONTENT_TYPES


# ---------- subject_entity validation ----------


@pytest.mark.parametrize(
    "good",
    [
        "sean-callagy",
        "sai-prime",
        "sai-forge",
        "kai",
        "aiko",
        "system",
        "unknown",
        "user:anna",
        "user:thomas-ferman",
        "user:phil",
        "case:cf-la-criminal",
        "case:cf-cra-audit-toronto",
    ],
)
def test_is_valid_subject_entity_accepts_canonical(good: str) -> None:
    assert is_valid_subject_entity(good)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "Sean Callagy",       # spaces / capitals
        "Phil",               # bare name without 'user:' prefix
        "user:Phil",          # capital in slug
        "user:",              # empty slug
        "user:phil!",         # disallowed punct
        "case:cf-",           # empty case slug after cf-
        "case:la-criminal",   # missing 'cf-' prefix
        "USER:anna",          # uppercase prefix
        "random-string",      # not in fixed set, not user:/case:
        None,
        42,
        [],
    ],
)
def test_is_valid_subject_entity_rejects_invalid(bad: object) -> None:
    assert not is_valid_subject_entity(bad)  # type: ignore[arg-type]


def test_normalize_user_slug() -> None:
    assert normalize_user_slug("Phil") == "phil"
    assert normalize_user_slug("MZea") == "mzea"
    assert normalize_user_slug("  thomas-ferman  ") == "thomas-ferman"
    assert normalize_user_slug("ALI") == "ali"


def test_known_user_slugs_are_real() -> None:
    """Smoke check: at least the top-3 by record count are present."""
    for slug in ("anna", "thomas-ferman", "ali"):
        assert slug in KNOWN_USER_SLUGS


# ---------- intents ----------


def test_intents_match_recall_context_tool_enum() -> None:
    """Mirrors library.py's RECALL_CONTEXT_TOOL.parameters.intent.enum."""
    expected = {
        "explain-teaching",
        "person-recall",
        "case-lookup",
        "kai-memory",
        "continuity-snapshot",
        "general",
    }
    assert INTENTS == expected


def test_intent_required_params() -> None:
    """Required params per intent — must mirror library.py:handle_recall_context."""
    assert INTENT_REQUIRED_PARAMS["person-recall"] == frozenset({"subject_entity"})
    assert INTENT_REQUIRED_PARAMS["continuity-snapshot"] == frozenset({"subject_entity"})
    assert INTENT_REQUIRED_PARAMS["case-lookup"] == frozenset({"case_id"})
    assert INTENT_REQUIRED_PARAMS["explain-teaching"] == frozenset()
    assert INTENT_REQUIRED_PARAMS["kai-memory"] == frozenset()
    assert INTENT_REQUIRED_PARAMS["general"] == frozenset()


def test_intent_required_params_covers_all_intents() -> None:
    assert set(INTENT_REQUIRED_PARAMS.keys()) == INTENTS


# ---------- topics ----------


def test_topics_includes_core_unblinded_concepts() -> None:
    for t in (
        "zone-action",
        "seven-levers",
        "heroic-unique-identity",
        "formula-cascade",
    ):
        assert t in TOPICS


# ---------- vendor-name forbidden list ----------


def test_forbidden_vendor_tokens_match_rubric() -> None:
    """Mirrors platform/eval/rubric.yaml identity_lockdown.forbidden_word_boundaries_ci."""
    expected = {
        "gpt", "claude", "llama", "mistral", "gemini", "qwen",
        "deepseek", "kimi", "moonshot", "anthropic", "openai",
    }
    assert FORBIDDEN_VENDOR_TOKENS == expected
