"""In-process rolling-summary memory replacing the n8n Postgres memory."""
from __future__ import annotations

import threading
from dataclasses import dataclass

_HEADER = "## Memory: prior chunk summaries"


@dataclass(frozen=True)
class _Entry:
    chunk_index: int
    summary: str

    def render(self) -> str:
        return f"[Section {self.chunk_index}]: {self.summary}"


class RollingMemory:
    """Append-only summary buffer with FIFO eviction by character budget.

    The orchestrator is expected to call ``add()`` in chunk-index order, but
    a lock is used so concurrent callers are still safe.
    """

    def __init__(self, max_chars: int) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        self._max_chars = max_chars
        self._entries: list[_Entry] = []
        self._lock = threading.Lock()

    def add(self, chunk_index: int, summary: str) -> None:
        with self._lock:
            self._entries.append(_Entry(chunk_index=chunk_index, summary=summary))

    def render(self) -> str:
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return ""

        # Newest entries are kept whole; if the very newest alone exceeds the
        # budget, truncate just that one to fit.
        kept: list[str] = []
        running = 0
        joiner = "\n"
        header = _HEADER
        # Reserve room for the header and the joiner that follows it.
        budget = self._max_chars - len(header) - len(joiner)

        if budget <= 0:
            # Header alone already exceeds budget — return a truncated header.
            return header[: self._max_chars]

        for entry in reversed(entries):
            line = entry.render()
            sep_cost = len(joiner) if kept else 0
            if running + sep_cost + len(line) <= budget:
                kept.append(line)
                running += sep_cost + len(line)
                continue
            if not kept:
                # Newest entry doesn't fit — truncate it to the available budget.
                kept.append(line[:budget])
                running = budget
            break

        body = joiner.join(reversed(kept))
        return f"{header}{joiner}{body}"
