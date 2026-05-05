"""Pydantic schemas — the contract between every module."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    chunk_index: int  # 1-based
    total_chunks: int
    position_context: str  # "[Section i of N]"
    session_id: str       # per-document, unique across runs


class RetrievedDoc(BaseModel):
    namespace: str
    score: float
    doc_id: str
    source_title: Optional[str] = None
    subject_entity: Optional[str] = None
    text: str


class Retrieval(BaseModel):
    primary: list[RetrievedDoc]      # from "teachings"
    supporting: list[RetrievedDoc]   # from cases/identity/etc.

    def render_for_prompt(self, max_chars_per_doc: int = 800) -> str:
        """Format retrieval into a prompt-ready block. Keeps token usage bounded."""
        lines: list[str] = []
        if self.primary:
            lines.append("## Retrieved teachings (primary)")
            for d in self.primary:
                title = d.source_title or d.doc_id
                lines.append(f"- [{title}] (score={d.score:.3f})\n{d.text[:max_chars_per_doc]}")
        if self.supporting:
            lines.append("\n## Supporting context")
            for d in self.supporting:
                title = d.source_title or d.doc_id
                lines.append(f"- [{d.namespace}/{title}] (score={d.score:.3f})\n{d.text[:max_chars_per_doc]}")
        return "\n\n".join(lines)


class TranslatedRow(BaseModel):
    """One row appended to the output sheet/CSV. Names match column headers exactly."""
    topic: str
    context: str
    formula_element: str
    main_lesson: str
    solves_what_human_condition: str
    seans_processing: str  # NB: no apostrophe, single-token name (no underscores in 'sean')
    seans_approach: str


class TranslationResult(BaseModel):
    """LLM returns this. Multiple rows allowed per chunk."""
    rows: list[TranslatedRow] = Field(min_length=1)
    chunk_summary: str = Field(
        description="≤200 word recap of what this chunk established — used as memory for later chunks."
    )


COLUMN_HEADERS: list[str] = [
    "topic",
    "context",
    "formula_element",
    "main_lesson",
    "solves_what_human_condition",
    "seans_processing",
    "seans_approach",
]
