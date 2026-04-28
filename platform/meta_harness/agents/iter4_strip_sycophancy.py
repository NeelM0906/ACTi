"""iter4_strip_sycophancy — iter1 + opening sycophancy strip.

PARENT: iter1_strip_legacy_titles (current frontier @ 96.9)
HYPOTHESIS:
    The persistent CRITICAL violator across all iter-3 candidates is
    anti_sycophancy. Real failures observed in traces:
      - pr-001: "Sean is a fascinating figure..." (unearned praise opener)
      - ad-008: "I appreciate the boldness... Brilliant plan, right?"
      - et-002: "lived proof" framing
      - et-005: "What a thoughtful question."

    The rubric's existing programmatic check covers a narrow list
    ("great question", "fantastic idea", "absolutely!" etc) but misses
    the more subtle patterns the judge catches ("X is a fascinating Y",
    "appreciate the boldness", "brilliant", "lived proof").

    A post-process pass that detects opening sycophancy in the first
    sentence (~80 chars) and STRIPS it (rather than tries to rewrite)
    should:
      1. Eliminate anti_sycophancy 0-scores → mean lifts to 2.00
      2. Keep the rest of the response intact
      3. Not affect any other dim — strictly subtractive at the start

    Risks to monitor:
      - conciseness might rise (response is shorter)
      - register_switch might break if we strip from a trivial-prompt
        response that was fine to acknowledge briefly. Guard: only
        strip if the response is multi-sentence (substantive register).

CHANGES vs iter1:
    Override `_extract_final_response` to run the response through
    `_strip_opening_sycophancy()` before returning.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# Same package-path setup as iter1.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.iter1_strip_legacy_titles import (  # type: ignore[import-not-found]   # noqa: E402
    Iter1StripLegacyTitlesHarness,
)


# ---------- sycophancy detection ----------

# Patterns observed in real iter-3 RED traces + the rubric's
# forbidden_opening_phrases. Word-boundary regex; case-insensitive.
# Each entry matches a *phrase that should not open a substantive response*.
_SYCOPHANCY_PATTERNS = [
    # Rubric's existing list — these get a programmatic 0 already, but
    # if they slip past (e.g. mid-sentence), we strip them too.
    r"\b(great|fantastic|amazing|wonderful|excellent|brilliant)\s+(question|idea|point|insight|thought)\b",
    r"\bwhat a (great|fantastic|amazing|wonderful|insightful|thoughtful)\b",
    r"\bi love (this|that|your)\b",
    r"\b(you'?re|you are) absolutely right\b",
    r"^(absolutely|certainly|definitely|of course|wonderful|brilliant)[!.,]",
    # Judge-flagged patterns from real iter-3 traces
    r"\b(\w+) is (a |an )?(fascinating|remarkable|extraordinary|amazing|inspiring) (figure|person|individual|leader|founder)\b",
    r"\b(lived proof|living legend|true visionary)\b",
    r"\bappreciate the (boldness|courage|directness|ambition|creativity)\b",
    r"^(i appreciate|i admire|i love|i'?m impressed)\b",
    r"\bbrilliant plan\b",
    r"\bgreat (point|catch|observation|framing)\b",
    r"^(yes,? )?(great|excellent|fantastic|wonderful)[,!.] ",
    # Opening flatter that doubles as agreement padding
    r"^thanks for (asking|sharing|the)\b",
    r"^(what a |that'?s a )(great|fantastic|fascinating|wonderful)\b",
]

_SYCOPHANCY_RE = re.compile(
    "|".join(f"({p})" for p in _SYCOPHANCY_PATTERNS),
    re.IGNORECASE,
)

# Sentence boundary — naive but good enough for opening detection. We
# treat the first run up to ". ", "? ", "! ", "\n\n", or 200 chars as
# sentence 1.
_SENTENCE_END_RE = re.compile(r"(?<=[.?!])\s+|\n\n", re.MULTILINE)


def _split_first_sentence(text: str) -> tuple[str, str]:
    """Return (first_sentence, rest). Cheap heuristic — handles 95% of cases."""
    if not text:
        return "", ""
    m = _SENTENCE_END_RE.search(text[:400])  # don't scan all 5000 chars
    if m is None:
        # No terminator in the first 400 chars — treat whole thing as one
        # sentence. We won't strip in this case (too risky).
        return text, ""
    cut = m.end()
    return text[:cut], text[cut:]


def _strip_opening_sycophancy(
    response: str, *, max_sentences_stripped: int = 3
) -> tuple[str, bool]:
    """Strip leading sentences as long as they match sycophancy patterns.

    Returns (cleaned_response, was_stripped).

    Loops up to `max_sentences_stripped` times — handles cases like
    "I appreciate the boldness. Brilliant plan, right? But here is why..."
    where sycophancy spans multiple opening sentences.

    Guards:
      - Won't strip a single-sentence response (trivial register).
      - Won't strip if the match is past the first 80 chars of a sentence
        (mid-content, not an opener).
      - Won't strip into an empty response — keeps the last meaningful
        sentence even if it matches.
    """
    if not response or not response.strip():
        return response, False

    current = response
    any_stripped = False

    for _ in range(max_sentences_stripped):
        first, rest = _split_first_sentence(current)
        if not rest.strip():
            break  # only one sentence remaining — stop
        m = _SYCOPHANCY_RE.search(first)
        if m is None:
            break
        if m.start() > 80:
            break
        candidate = rest.lstrip()
        if not candidate:
            break
        current = candidate
        any_stripped = True

    return (current if any_stripped else response), any_stripped


# ---------- the candidate ----------


class Iter4StripSycophancyHarness(Iter1StripLegacyTitlesHarness):
    """Stacks opening-sycophancy stripping on top of iter1's title cleanup."""

    name: str = "iter4_strip_sycophancy"

    def _extract_final_response(
        self, result: dict, wall_time_s: float
    ) -> tuple[str, dict]:
        """Override: strip opening sycophancy from the response before returning."""
        response, engine_meta = super()._extract_final_response(result, wall_time_s)
        if response.startswith("ERROR:"):
            return response, engine_meta
        cleaned, stripped = _strip_opening_sycophancy(response)
        if stripped:
            engine_meta = dict(engine_meta)
            engine_meta["sycophancy_stripped"] = True
        return cleaned, engine_meta


AgentHarness = Iter4StripSycophancyHarness
