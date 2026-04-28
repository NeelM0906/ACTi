"""Tests for lib/citations.py — chunk_id parsing + citation resolution."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the parent package importable when run from tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.citations import (  # noqa: E402
    ChunkId,
    extract_citations,
    parse_chunk_id,
    resolve_citation,
    validate_response_citations,
)


# ---------- parse_chunk_id ----------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "teaching:sean-callagy:115f5307444a:0000",
            ChunkId("teaching", "sean-callagy", "115f5307444a", 0),
        ),
        (
            "user-note:thomas-ferman:d89a6ad84f84:0001",
            ChunkId("user-note", "thomas-ferman", "d89a6ad84f84", 1),
        ),
        (
            "memory-daily:kai:900de4e65fbe:0042",
            ChunkId("memory-daily", "kai", "900de4e65fbe", 42),
        ),
        (
            "case-folder:cf-cra-audit-toronto:76261445bdd9:0000",
            ChunkId("case-folder", "cf-cra-audit-toronto", "76261445bdd9", 0),
        ),
        (
            "identity:sai-prime:376bed61c224:0000",
            ChunkId("identity", "sai-prime", "376bed61c224", 0),
        ),
    ],
)
def test_parse_chunk_id_real_examples(raw: str, expected: ChunkId) -> None:
    """All five real chunk_ids from the canonical store + sm_sample_queries.json."""
    assert parse_chunk_id(raw) == expected


def test_parse_chunk_id_round_trips() -> None:
    """str(ChunkId) reproduces the input exactly."""
    raw = "teaching:sean-callagy:115f5307444a:0042"
    parsed = parse_chunk_id(raw)
    assert parsed is not None
    assert str(parsed) == raw


@pytest.mark.parametrize(
    "bad",
    [
        "",                                              # empty
        "teaching",                                      # too few parts
        "teaching:sean-callagy:115f5307444a",            # missing chunk_index
        "teaching:sean-callagy:SHORT:0000",              # hash too short
        "teaching:sean-callagy:115f5307444aBAD:0000",    # non-hex / too long
        "teaching:sean-callagy:115f5307444a:42",         # chunk_index not 4 digits
        "TEACHING:sean-callagy:115f5307444a:0000",       # uppercase content_type
        "fake-type:sean-callagy:115f5307444a:0000",      # unknown content_type
        "teaching:Phil!:115f5307444a:0000",              # disallowed slug char
        "DEN-FL-001",                                    # tag-prefix shorthand, not chunk_id
    ],
)
def test_parse_chunk_id_rejects_malformed(bad: str) -> None:
    assert parse_chunk_id(bad) is None


def test_parse_chunk_id_rejects_non_string() -> None:
    """The parser tolerates wrong types instead of raising."""
    for bad in (None, 42, [], {}, b"teaching:x:111111111111:0000"):
        assert parse_chunk_id(bad) is None  # type: ignore[arg-type]


# ---------- extract_citations ----------


def test_extract_citations_basic() -> None:
    text = (
        "From [teaching:sean-callagy:115f5307444a:0000] Sean defines Zone Action "
        "as a non-redundant action. Also see [DEN-FL-001] for case context."
    )
    assert extract_citations(text) == [
        "teaching:sean-callagy:115f5307444a:0000",
        "DEN-FL-001",
    ]


def test_extract_citations_skips_markdown_links() -> None:
    """Markdown link text [foo](url) should not be treated as a citation."""
    text = "See [the docs](https://example.com) and [teaching:x:111111111111:0000]."
    assert extract_citations(text) == ["teaching:x:111111111111:0000"]


def test_extract_citations_skips_trivial_brackets() -> None:
    """Single-token brackets like [1] / [*] / [ ] / [a] are not citations."""
    text = "Step [1] then [2] then [3]. Footnote [*]. Empty [ ]. Real [DEN-FL-001]."
    out = extract_citations(text)
    # 'a' is single alnum char (length 1 stripped), should be skipped too
    assert out == ["DEN-FL-001"]


def test_extract_citations_skips_multiline_brackets() -> None:
    """A bracket run that spans a newline isn't a citation."""
    text = "[ open\n close ] not a cite"
    assert extract_citations(text) == []


def test_extract_citations_empty_input() -> None:
    assert extract_citations("") == []
    assert extract_citations(None) == []  # type: ignore[arg-type]


# ---------- resolve_citation ----------


def _hit(
    chunk_id: str,
    *,
    source_title: str = "",
    namespace: str = "teachings",
    score: float = 0.5,
) -> dict:
    """Helper: build a hit dict in the shape the retrieval service returns."""
    return {
        "id": chunk_id,
        "score": score,
        "metadata": {
            "source_title": source_title,
            "namespace": namespace,
            "chunk_id": chunk_id,
        },
    }


def test_resolve_full_chunk_id_match() -> None:
    cid = "teaching:sean-callagy:115f5307444a:0000"
    hits = [_hit(cid, source_title="Sean teaching on Zone Action")]
    r = resolve_citation(cid, hits)
    assert r.valid is True
    assert r.form == "full"
    assert r.resolved_to_hit_id == cid


def test_resolve_full_chunk_id_no_match_is_fabrication() -> None:
    """A chunk_id-shaped citation that doesn't appear in hits is a fabrication."""
    cid = "teaching:sean-callagy:115f5307444a:0000"
    other = "teaching:kai:0123456789ab:0000"
    hits = [_hit(other)]
    r = resolve_citation(cid, hits)
    assert r.valid is False
    assert r.form == "unknown"
    assert r.resolved_to_hit_id is None


def test_resolve_tag_prefix_match() -> None:
    cid = "case-folder:cf-denver-family-law:abcdef012345:0001"
    hits = [_hit(cid, source_title="[DEN-FL-001] Apex Architect - Production Manifest")]
    r = resolve_citation("DEN-FL-001", hits)
    assert r.valid is True
    assert r.form == "tag-prefix"
    assert r.resolved_to_hit_id == cid


def test_resolve_tag_prefix_case_insensitive() -> None:
    cid = "case-folder:cf-denver-family-law:abcdef012345:0001"
    hits = [_hit(cid, source_title="[DEN-FL-001] Manifest")]
    r = resolve_citation("den-fl-001", hits)
    assert r.valid is True
    assert r.form == "tag-prefix"


def test_resolve_source_title_match() -> None:
    cid = "teaching:sean-callagy:115f5307444a:0000"
    hits = [_hit(cid, source_title="Apex Architect - Production Manifest")]
    r = resolve_citation("Apex Architect - Production Manifest", hits)
    assert r.valid is True
    assert r.form == "source-title"


def test_resolve_source_title_after_tag_prefix() -> None:
    """If source_title is '[TAG] Rest', citing 'Rest' should also resolve."""
    cid = "case-folder:cf-x:abcdef012345:0000"
    hits = [_hit(cid, source_title="[XYZ-001] Apex Architect - Manifest")]
    r = resolve_citation("Apex Architect - Manifest", hits)
    assert r.valid is True
    assert r.form == "source-title"


def test_resolve_unknown_returns_invalid() -> None:
    hits = [_hit("teaching:sean-callagy:111111111111:0000", source_title="Real Title")]
    r = resolve_citation("Some Made-Up Title", hits)
    assert r.valid is False
    assert r.form == "unknown"


# ---------- validate_response_citations ----------


def test_validate_response_full_pipeline() -> None:
    """One real cite + one fabricated cite → 1 valid, 1 invalid."""
    real_id = "teaching:sean-callagy:115f5307444a:0000"
    fake_id = "teaching:fabricated:000000000000:0000"
    hits = [_hit(real_id, source_title="Real teaching")]

    response = (
        f"Per [{real_id}] Sean defines Zone Action as non-redundant action. "
        f"And per [{fake_id}] there's also a Pareto-39-Elements concept."
    )
    out = validate_response_citations(response, hits)
    assert len(out) == 2
    assert out[0].valid is True and out[0].form == "full"
    assert out[1].valid is False and out[1].form == "unknown"


def test_validate_response_no_citations() -> None:
    assert validate_response_citations("Plain text, no brackets at all.", []) == []


def test_validate_response_skips_markdown_links() -> None:
    """The markdown link should not count as a citation."""
    out = validate_response_citations("See [docs](https://x.com) for details.", [])
    assert out == []


def test_validate_response_handles_no_hits() -> None:
    """If a response cites anything when no hits were returned, all are invalid."""
    response = "Per [teaching:sean-callagy:111111111111:0000] this and that."
    out = validate_response_citations(response, [])
    assert len(out) == 1
    assert out[0].valid is False
