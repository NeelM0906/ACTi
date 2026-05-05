"""PDF text extraction. Single entry point used by the orchestrator."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pdfplumber

log = logging.getLogger(__name__)

_WS_BEFORE_NL = re.compile(r"\s+\n")


def extract_pdf_text(path: str | Path) -> tuple[str, str]:
    """Extract text from a PDF and derive a title from the filename.

    Returns ``(extracted_text, derived_title)``. Raises ``ValueError`` if the
    PDF yields no text after extraction.
    """
    p = Path(path)
    pages: list[str] = []
    with pdfplumber.open(p) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

    raw = "\n\n".join(pages)
    # Mirror the n8n JS preprocessing: collapse trailing whitespace before newlines.
    normalized = _WS_BEFORE_NL.sub("\n", raw).strip()

    if not normalized:
        raise ValueError(f"No extractable text in PDF: {p}")

    title = _derive_title(p.name)
    log.debug("extracted %d chars from %s (title=%r)", len(normalized), p, title)
    return normalized, title


def _derive_title(filename: str) -> str:
    stem = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    cleaned = stem.replace("_", " ").strip()
    return cleaned or "Untitled"
