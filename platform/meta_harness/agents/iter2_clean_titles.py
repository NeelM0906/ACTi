"""iter2_clean_titles — extend iter1 to clean ALL source_titles, not just placeholders.

PARENT: iter1_strip_legacy_titles
HYPOTHESIS:
    iter1 lifted citation_grounding marginally (+0.02) but introduced a
    no_emoji regression: real-corpus source_titles can contain emoji
    (Kai journal entries like "## 🕙 10:00 PM — Memory Sync"), and the
    model copies them verbatim into citations.

    A complete title-cleaning pass must:
      1. Rewrite placeholder titles to semantic labels (iter1 behavior).
      2. Strip emoji codepoints from EVERY title.
      3. Collapse whitespace runs.
      4. Cap length at 80 chars (no model needs a 200-char citation tag).

    Predicted effect:
      - Same cl-003 win as iter1 (placeholder fix)
      - Restore no_emoji to 2.00 (no emoji can leak via copied titles)
      - Possibly small lift in citation_grounding (cleaner cite tags)

CHANGES vs iter1:
    Replace `_clean_legacy_source_titles` with `_clean_all_source_titles`
    that runs on every hit, applying the four rules above. Inherit the
    rest of the iter1 pipeline unchanged.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

# Same package-path setup as iter1.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.iter1_strip_legacy_titles import (  # type: ignore[import-not-found]   # noqa: E402
    Iter1StripLegacyTitlesHarness,
    _PLACEHOLDER_RE,
    _semantic_title,
)
import library  # type: ignore[import-not-found]   # noqa: E402
from lib.harness import ToolCallTrace  # type: ignore[import-not-found]   # noqa: E402

# ---------- emoji stripping ----------

# Match every codepoint range that the no_emoji rubric programmatic check
# considers an emoji (rubric.yaml line 90-98). Kept in sync deliberately;
# any codepoint that scores 0 on no_emoji must be stripped here.
_EMOJI_RANGES = [
    (0x1F300, 0x1FAFF),
    (0x2600,  0x27BF),
    (0x1F000, 0x1F2FF),
    (0x1F600, 0x1F64F),
    (0x2300,  0x23FF),
    (0xFE0F,  0xFE0F),
]


def _strip_emoji(s: str) -> str:
    return "".join(
        ch for ch in s
        if not any(lo <= ord(ch) <= hi for lo, hi in _EMOJI_RANGES)
    )


_WHITESPACE_RE = re.compile(r"\s+")
_MAX_TITLE_CHARS = 80


def _clean_one_title(metadata: dict[str, Any]) -> str:
    """Apply all four cleanup rules. Returns the cleaned title."""
    src = (metadata.get("source_title") or "").strip()
    # 1. Placeholder rewrite (iter1 behavior).
    if not src or _PLACEHOLDER_RE.search(src):
        src = _semantic_title(metadata) or "Unblinded corpus chunk"
    # 2. Strip emoji.
    src = _strip_emoji(src)
    # 3. Collapse whitespace.
    src = _WHITESPACE_RE.sub(" ", src).strip()
    # 4. Cap length.
    if len(src) > _MAX_TITLE_CHARS:
        src = src[: _MAX_TITLE_CHARS - 1].rstrip() + "…"
    if not src:
        # If everything got stripped, fall back to a generic label.
        src = (
            _semantic_title(metadata)
            or f"{metadata.get('namespace', 'corpus')} chunk"
        )
    return src


def _clean_all_source_titles(hits: list[dict]) -> list[dict]:
    """Apply _clean_one_title to every hit. Idempotent."""
    out: list[dict] = []
    for h in hits:
        md = h.get("metadata") or {}
        new_title = _clean_one_title(md)
        if new_title == md.get("source_title"):
            out.append(h)
            continue
        new_md = dict(md)
        new_md["_original_source_title"] = md.get("source_title", "")
        new_md["source_title"] = new_title
        new_h = dict(h)
        new_h["metadata"] = new_md
        out.append(new_h)
    return out


# ---------- the candidate ----------


class Iter2CleanTitlesHarness(Iter1StripLegacyTitlesHarness):
    """Extends iter1 with full title cleanup (emoji + whitespace + length)."""

    name: str = "iter2_clean_titles"

    def _make_capturing_handler(
        self,
        captured_calls: list[ToolCallTrace],
        captured_hits_flat: list[dict],
        captured_meta: dict[str, Any],
    ):
        """Override: use _clean_all_source_titles instead of placeholder-only."""

        async def _handler(args: dict) -> str:
            t0 = time.time()
            turn_idx = sum(
                1 for c in captured_calls if c["name"] == "recall_context"
            ) + 1

            # --- mirror baseline validation block (same code path as iter1) ---
            if not library.library_enabled():
                msg = "ERROR: the Unblinded library is not configured on this deployment."
                captured_calls.append(self._mk_call_trace(
                    turn_idx, args, msg, [], int((time.time() - t0) * 1000)
                ))
                return msg
            query = (args.get("query") or "").strip()
            if not query:
                msg = "ERROR: `query` is required."
                captured_calls.append(self._mk_call_trace(
                    turn_idx, args, msg, [], int((time.time() - t0) * 1000)
                ))
                return msg
            intent = args.get("intent") or "general"
            if (
                intent in {"person-recall", "continuity-snapshot"}
                and not args.get("subject_entity")
            ):
                msg = (
                    f"ERROR: intent={intent!r} requires `subject_entity` "
                    f"(format: `user:<slug>`, e.g. `user:adam-gugino`). "
                    f"Either provide it, or call with intent=general."
                )
                captured_calls.append(self._mk_call_trace(
                    turn_idx, args, msg, [], int((time.time() - t0) * 1000)
                ))
                return msg
            if intent == "case-lookup" and not args.get("case_id"):
                msg = (
                    f"ERROR: intent='case-lookup' requires `case_id` "
                    f"(format: `cf-<slug>`, e.g. `cf-denver-family-law`). "
                    f"Either provide it, or call with intent=general."
                )
                captured_calls.append(self._mk_call_trace(
                    turn_idx, args, msg, [], int((time.time() - t0) * 1000)
                ))
                return msg

            payload: dict[str, Any] = {"query": query, "intent": intent}
            if args.get("subject_entity"):
                payload["subject_entity"] = args["subject_entity"]
            if args.get("case_id"):
                payload["case_id"] = args["case_id"]
            if args.get("top_k"):
                payload["top_k"] = max(
                    1, min(int(args["top_k"]), library.LIBRARY_MAX_HITS)
                )
            else:
                payload["top_k"] = library.LIBRARY_MAX_HITS

            result = await library._call_retrieve(payload)
            duration_ms = int((time.time() - t0) * 1000)

            if "error" in result:
                msg = f"ERROR: {result['error']}"
                captured_calls.append(self._mk_call_trace(
                    turn_idx, args, msg, [], duration_ms
                ))
                return msg

            hits_raw = result.get("hits") or []
            # --- THE iter2 CHANGE: clean ALL titles, not just placeholders ---
            hits = _clean_all_source_titles(hits_raw)
            # --- end change ---

            captured_hits_flat.extend(hits)
            captured_meta.setdefault("intents_used", []).append(intent)
            captured_meta["last_intent"] = intent
            captured_meta["last_payload"] = payload
            n_changed = sum(
                1 for h in hits
                if (h.get("metadata") or {}).get("_original_source_title")
                and (h.get("metadata") or {}).get("_original_source_title")
                != (h.get("metadata") or {}).get("source_title")
            )
            captured_meta["titles_cleaned"] = (
                captured_meta.get("titles_cleaned", 0) + n_changed
            )

            formatted = library._format_hits(intent, query, hits)
            captured_calls.append(
                self._mk_call_trace(turn_idx, args, formatted, hits, duration_ms)
            )
            return formatted

        return _handler


AgentHarness = Iter2CleanTitlesHarness
