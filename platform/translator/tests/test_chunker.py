"""Tests for sohn_translator.chunker."""
from __future__ import annotations

import pytest

from sohn_translator.chunker import chunk_text
from sohn_translator.schema import Chunk


@pytest.mark.unit
def test_empty_text_returns_empty_list() -> None:
    assert chunk_text("", session_id="s", chunk_size=100, overlap=10) == []


@pytest.mark.unit
def test_single_chunk_when_text_under_chunk_size() -> None:
    text = "Short body of text."
    chunks = chunk_text(text, session_id="sess-1", chunk_size=100, overlap=10)

    assert len(chunks) == 1
    only = chunks[0]
    assert isinstance(only, Chunk)
    assert only.text == text
    assert only.chunk_index == 1
    assert only.total_chunks == 1
    assert only.position_context == "[Section 1 of 1]"
    assert only.session_id == "sess-1"


@pytest.mark.unit
def test_multi_chunk_with_clean_paragraph_boundaries() -> None:
    # Each "para" is ~80 chars; chunk_size=200 → boundary at \n\n is past 60%.
    para = "A" * 80
    text = "\n\n".join([para, para, para, para])  # 4 paragraphs

    chunks = chunk_text(text, session_id="doc-x", chunk_size=200, overlap=10)

    assert len(chunks) >= 2
    assert all(c.total_chunks == len(chunks) for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(1, len(chunks) + 1))
    assert chunks[0].position_context == f"[Section 1 of {len(chunks)}]"
    assert chunks[-1].position_context == f"[Section {len(chunks)} of {len(chunks)}]"
    # First chunk should end on a paragraph boundary, not mid-letter.
    assert chunks[0].text.endswith("\n\n") or chunks[0].text.endswith(para)


@pytest.mark.unit
def test_hard_cut_when_no_good_boundary_exists() -> None:
    # No separators at all → must hard-cut at chunk_size.
    text = "X" * 500
    chunks = chunk_text(text, session_id="s", chunk_size=100, overlap=10)

    assert len(chunks) > 1
    # First chunk is exactly chunk_size (hard cut).
    assert len(chunks[0].text) == 100


@pytest.mark.unit
def test_overlap_is_observed_between_consecutive_chunks() -> None:
    text = "Y" * 1000
    chunk_size = 200
    overlap = 50
    chunks = chunk_text(text, session_id="s", chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:]):
        suffix = prev.text[-overlap:]
        prefix = nxt.text[:overlap]
        assert suffix == prefix


@pytest.mark.unit
def test_boundary_below_60_percent_is_ignored() -> None:
    # Sentence boundary lives at char ~20 of a 100-size window — below the 60%
    # threshold, so the chunker must hard-cut instead of honoring it.
    head = "Tiny. " + ("Z" * 200)
    chunks = chunk_text(head, session_id="s", chunk_size=100, overlap=0)

    assert chunks[0].text.startswith("Tiny. ")
    # Hard cut → first chunk should be exactly 100 chars, not the 6-char "Tiny. ".
    assert len(chunks[0].text) == 100


@pytest.mark.unit
def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", session_id="s", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("abc", session_id="s", chunk_size=10, overlap=-1)


@pytest.mark.unit
def test_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", session_id="s", chunk_size=0, overlap=0)
