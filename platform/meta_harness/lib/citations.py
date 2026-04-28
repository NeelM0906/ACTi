"""Citation parsing + validation for the ACTi corpus.

The canonical chunk_id grammar (from pinecone-audit/plans/target-schema.md §3):

    chunk_id = content_type ":" entity_slug ":" doc_hash12 ":" chunk_index4

Concrete examples we've seen in real hits:

    teaching:sean-callagy:115f5307444a:0000
    user-note:thomas-ferman:d89a6ad84f84:0000
    memory-daily:kai:900de4e65fbe:0000
    case-folder:cf-cra-audit-toronto:76261445bdd9:0000
    identity:sai-prime:376bed61c224:0000

Bracketed citations in Sohn's responses appear in three valid forms:

    1. Full chunk_id:        [teaching:sean-callagy:115f5307444a:0000]
    2. Tag-prefix shorthand: [DEN-FL-001]   (matches a hit whose source_title
                                             starts with "[DEN-FL-001] ...")
    3. source_title text:    [Apex Architect - Production Manifest]

The harness uses this module to:
    - extract every bracketed citation Sohn emitted,
    - resolve each one against the set of hits actually returned by the
      retrieval call, and
    - report which citations are valid / which were fabricated.

A response that cites a chunk_id NOT in any returned hit is the canonical
"fabrication" failure — programmatically detectable here, no judge needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

# ---------- chunk_id grammar ----------

CONTENT_TYPES: frozenset[str] = frozenset({
    "teaching",
    "transcript",
    "memory-daily",
    "memory-longterm",
    "user-note",
    "case-folder",
    "product-doc",
    "identity",
    "protocol",
    "other",
})

# Per target-schema.md §3.3:
#   entity_slug : 1*64 (ALPHA / DIGIT / "-" / "." / "_")
#   doc_hash12  : 12 lowercase hex chars
#   chunk_index : 4 digits (zero-padded)
_CHUNK_ID_RE = re.compile(
    r"^(?P<content_type>[a-z][a-z0-9-]*?)"
    r":(?P<entity_slug>[A-Za-z0-9._-]{1,64})"
    r":(?P<doc_hash>[a-f0-9]{12})"
    r":(?P<chunk_index>\d{4})$"
)


@dataclass(frozen=True)
class ChunkId:
    """Parsed chunk_id components. Round-trips via str()."""

    content_type: str
    entity_slug: str
    doc_hash: str
    chunk_index: int  # 0..9999

    def __str__(self) -> str:
        return (
            f"{self.content_type}:{self.entity_slug}:"
            f"{self.doc_hash}:{self.chunk_index:04d}"
        )


def parse_chunk_id(s: str) -> ChunkId | None:
    """Parse a chunk_id string. Return None if it doesn't match the grammar.

    Strict — the content_type must be in CONTENT_TYPES. Returns None for
    anything malformed, including tag-prefix shorthands like "DEN-FL-001"
    (those are NOT chunk_ids; they're hit-title prefixes).
    """
    if not isinstance(s, str):
        return None
    m = _CHUNK_ID_RE.match(s.strip())
    if m is None:
        return None
    if m.group("content_type") not in CONTENT_TYPES:
        return None
    return ChunkId(
        content_type=m.group("content_type"),
        entity_slug=m.group("entity_slug"),
        doc_hash=m.group("doc_hash"),
        chunk_index=int(m.group("chunk_index")),
    )


# ---------- citation extraction ----------

# Bracketed run: [...] where the body is non-empty and contains no nested
# brackets. We DO NOT strip whitespace inside the brackets — a citation
# like "[ Apex Architect - X ]" is a different (and probably wrong)
# rendering than "[Apex Architect - X]".
_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def extract_citations(text: str) -> list[str]:
    """Return every bracketed-citation body in `text`, in order of appearance.

    Returns the BODY (without the surrounding brackets). Duplicates are
    preserved — caller decides whether to dedupe.

    Skips obviously-not-a-citation matches:
      - bodies that look like markdown link text (followed by `(...)`)
      - bodies that are literally a single number or punctuation token
      - bodies that contain a newline (multi-line bracket runs are rare
        and almost certainly non-citation content)
    """
    if not isinstance(text, str) or not text:
        return []
    out: list[str] = []
    for m in _CITATION_RE.finditer(text):
        body = m.group(1)
        # Skip obvious non-citations.
        if "\n" in body:
            continue
        # Skip markdown-style [text](url): peek at next char.
        end = m.end()
        if end < len(text) and text[end] == "(":
            continue
        # Skip single tokens like `[1]`, `[ ]`, `[*]` — not useful as cites.
        if len(body.strip()) < 2 or not any(c.isalnum() for c in body):
            continue
        out.append(body)
    return out


# ---------- citation resolution against returned hits ----------

CitationForm = Literal["full", "tag-prefix", "source-title", "unknown"]


@dataclass(frozen=True)
class ResolvedCitation:
    """Outcome of resolving one bracketed citation against returned hits."""

    raw: str  # the body, exactly as it appeared between [ and ]
    form: CitationForm  # how it matched (or 'unknown' if no match)
    valid: bool  # True iff form != 'unknown'
    resolved_to_hit_id: str | None = None


def _hit_id(hit: dict) -> str:
    """Extract the chunk_id from a retrieval hit. Hits use either
    `id` (top-level) or fall back to metadata.chunk_id.
    """
    return str(hit.get("id") or hit.get("metadata", {}).get("chunk_id") or "")


def _hit_source_title(hit: dict) -> str:
    return str((hit.get("metadata") or {}).get("source_title") or "")


def _tag_prefix(source_title: str) -> str | None:
    """If a source_title begins with "[TAG] rest...", return TAG. Else None.

    Used by the proposer to allow tag-prefix shorthand citation
    (per platform/eval/rubric.yaml citation_grounding §270-274).
    """
    if not source_title.startswith("["):
        return None
    end = source_title.find("]")
    if end <= 1:
        return None
    return source_title[1:end].strip() or None


def resolve_citation(body: str, hits: Iterable[dict]) -> ResolvedCitation:
    """Try to resolve one citation body against the returned hits.

    Resolution order:
      1. Exact chunk_id match (form="full").
      2. Tag-prefix match: body equals the [TAG] prefix of any hit's
         source_title (form="tag-prefix").
      3. Source_title equality, case-insensitive after stripping any
         leading "[TAG] " (form="source-title").

    No partial / fuzzy matching. The grammar is deterministic by design.
    """
    body_stripped = body.strip()
    hits_list = list(hits)

    # 1. Full chunk_id?
    parsed = parse_chunk_id(body_stripped)
    if parsed is not None:
        target = str(parsed)
        for h in hits_list:
            if _hit_id(h) == target:
                return ResolvedCitation(
                    raw=body, form="full", valid=True, resolved_to_hit_id=target
                )
        # Looks like a chunk_id but no hit matches → fabrication.
        return ResolvedCitation(raw=body, form="unknown", valid=False)

    # 2. Tag-prefix shorthand?
    for h in hits_list:
        st = _hit_source_title(h)
        tag = _tag_prefix(st)
        if tag and tag.casefold() == body_stripped.casefold():
            return ResolvedCitation(
                raw=body, form="tag-prefix", valid=True, resolved_to_hit_id=_hit_id(h)
            )

    # 3. Source-title equality.
    target_cf = body_stripped.casefold()
    for h in hits_list:
        st = _hit_source_title(h)
        # Compare against the full source_title and against its post-[TAG] tail.
        if st.casefold() == target_cf:
            return ResolvedCitation(
                raw=body, form="source-title", valid=True, resolved_to_hit_id=_hit_id(h)
            )
        tag = _tag_prefix(st)
        if tag is not None:
            tail = st[len(tag) + 2 :].lstrip()  # +2 for the surrounding [ ]
            if tail.casefold() == target_cf:
                return ResolvedCitation(
                    raw=body,
                    form="source-title",
                    valid=True,
                    resolved_to_hit_id=_hit_id(h),
                )

    return ResolvedCitation(raw=body, form="unknown", valid=False)


def validate_response_citations(
    response_text: str, hits: list[dict]
) -> list[ResolvedCitation]:
    """High-level helper: extract every bracketed citation and resolve each.

    Returns an in-order list. A scenario response that emits no citations
    returns []. Use the `valid` field on each entry to compute precision.
    """
    bodies = extract_citations(response_text)
    return [resolve_citation(b, hits) for b in bodies]
