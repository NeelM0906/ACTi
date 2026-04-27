"""Tests for spark._sanitize_tool_call_args.

Covers the XML-tag-based parser-leak case where the engine's tool-call parser
extracts a parameter value as everything between `<parameter=name>` and
`</parameter>`, and a stray `>` emitted by the model leaks into the value.

Run:
    python -m pytest platform/proxy/test_tool_call_sanitizer.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from spark import _sanitize_tool_call_args


def _roundtrip(d: dict) -> dict:
    out = _sanitize_tool_call_args(json.dumps(d))
    return json.loads(out)


def test_strips_trailing_gt_from_intent():
    """The exact bug from eval cs-003."""
    assert _roundtrip({
        "intent": "continuity-snapshot>",
        "subject_entity": "Kai",
    }) == {"intent": "continuity-snapshot", "subject_entity": "Kai"}


def test_strips_trailing_lt_too():
    """If the parser leaks an opening tag start, strip that too."""
    assert _roundtrip({"query": "Kai journal<"}) == {"query": "Kai journal"}


def test_strips_trailing_whitespace_around_angle_brackets():
    """Whitespace between value and the leaked bracket should also go."""
    assert _roundtrip({"intent": "continuity-snapshot >  "}) == {"intent": "continuity-snapshot"}


def test_only_strips_trailing_brackets():
    """Brackets in the middle of a string are content, not parser leaks."""
    assert _roundtrip({"query": "what is <Zone Action>"}) == {"query": "what is <Zone Action>"}


def test_preserves_non_string_values():
    """Numbers, bools, nulls, lists, dicts pass through untouched."""
    src = {
        "top_k": 8,
        "rerank": True,
        "filter": None,
        "tags": ["a", "b"],
        "nested": {"x": 1},
    }
    assert _roundtrip(src) == src


def test_preserves_clean_string_values():
    """A string with no trailing junk is unchanged."""
    src = {"query": "Sean Callagy", "intent": "person-recall"}
    assert _roundtrip(src) == src


def test_handles_empty_string():
    """Empty arguments — passthrough, no exception."""
    assert _sanitize_tool_call_args("") == ""
    assert _sanitize_tool_call_args("{}") == "{}"


def test_handles_malformed_json_gracefully():
    """If the JSON itself is broken, return as-is (caller will surface the error)."""
    bad = '{"intent": "x", "broken'
    assert _sanitize_tool_call_args(bad) == bad


def test_handles_non_dict_top_level():
    """Top-level array/scalar JSON is not what tool args are; passthrough."""
    assert _sanitize_tool_call_args("[]") == "[]"
    assert _sanitize_tool_call_args('"hello"') == '"hello"'


def test_strips_multiple_fields_at_once():
    """Two leaked fields in one args object — both get fixed."""
    assert _roundtrip({
        "intent": "continuity-snapshot>",
        "subject_entity": "user:kai>",
    }) == {"intent": "continuity-snapshot", "subject_entity": "user:kai"}


def test_idempotent():
    """Running the sanitizer twice on the same input is a no-op the second time."""
    once = _sanitize_tool_call_args(json.dumps({"intent": "continuity-snapshot>"}))
    twice = _sanitize_tool_call_args(once)
    assert once == twice


def test_no_change_returns_original_string():
    """When nothing needs cleaning, the function returns the input verbatim
    (no JSON re-serialization that could reorder keys or change spacing)."""
    src = '{"query": "x", "intent": "person-recall"}'
    assert _sanitize_tool_call_args(src) == src
