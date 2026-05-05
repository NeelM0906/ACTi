"""Boundary-aware text chunker mirroring the original n8n Doc node."""
from __future__ import annotations

import logging

from .schema import Chunk

log = logging.getLogger(__name__)

# Boundary patterns in priority order. The original n8n Doc node prefers
# paragraph breaks, then line breaks, then sentence terminators.
_BOUNDARIES: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ")

# Only honor a boundary if it falls past this fraction of chunk_size.
_MIN_BOUNDARY_RATIO = 0.6


def chunk_text(
    text: str,
    *,
    session_id: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    if not text:
        return []

    raw_chunks = _split_with_boundaries(text, chunk_size=chunk_size, overlap=overlap)
    total = len(raw_chunks)
    return [
        Chunk(
            text=body,
            chunk_index=i + 1,
            total_chunks=total,
            position_context=f"[Section {i + 1} of {total}]",
            session_id=session_id,
        )
        for i, body in enumerate(raw_chunks)
    ]


def _split_with_boundaries(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    n = len(text)
    if n <= chunk_size:
        return [text]

    min_cut = int(chunk_size * _MIN_BOUNDARY_RATIO)
    chunks: list[str] = []
    start = 0

    while start < n:
        # Last chunk: take the remainder verbatim.
        if start + chunk_size >= n:
            chunks.append(text[start:])
            break

        window_end = start + chunk_size
        cut = _find_boundary(text, start=start, window_end=window_end, min_cut=min_cut)
        if cut is None:
            cut = window_end  # hard cut

        chunks.append(text[start:cut])

        # Step forward, leaving `overlap` chars of context for the next chunk.
        next_start = cut - overlap
        if next_start <= start:
            # Pathological: ensure forward progress.
            next_start = cut
        start = next_start

    return chunks


def _find_boundary(text: str, *, start: int, window_end: int, min_cut: int) -> int | None:
    """Find the best boundary within [start+min_cut, window_end].

    Walks the separators in priority order and returns the first one whose
    rightmost occurrence lies in the window. Returns the absolute index
    *just after* the boundary token (so slicing ``text[start:cut]`` keeps
    the boundary). ``None`` means no boundary qualifies — caller must hard-cut.
    """
    earliest = start + min_cut
    for sep in _BOUNDARIES:
        idx = text.rfind(sep, earliest, window_end)
        if idx != -1:
            return idx + len(sep)
    return None
