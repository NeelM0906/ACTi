"""iter5_attribution_recovery — recover subject_entity for "unknown" hits.

PARENT: iter1_strip_legacy_titles (current frontier)
HYPOTHESIS:
    The corpus audit (~/Desktop/ACTi_base/pinecone-audit/) showed that
    24,624 of 25,052 records in the `users` namespace have
    `subject_entity = "unknown"` — a 98.3% attribution gap that is the
    single largest data-quality issue in the corpus.

    When a person-recall query lands on these unknown-attributed hits,
    the model can't filter by person and can't cite specifically. The
    judge frequently flags responses as ungrounded because the
    namespace=users hit's subject_entity is unknown — even though the
    text content does identify a specific person.

    Recovery is feasible from three signals already present in the hit:
      1. source.uri — pattern `legacy://saimemory/<slug>/...` carries
         the original namespace name, which was almost always a slug.
      2. source.title — strings like "saimemory/<slug> (legacy)" carry
         the slug.
      3. text body — first 200 chars often contain a known user slug
         (anna, phil, ali, mzea, mj, jared, miko, thomas-ferman, members)
         in the first sentence.

    Recovered slugs get tagged onto the hit metadata so the formatted
    output (which the model sees) reflects the real person, AND the
    citations emitted by the model land on a meaningful subject_entity
    rather than "unknown".

    Predicted effect:
      - person-recall (pr-*) intent mean: 94.0 → ≥96
      - citation_grounding: stable or up slightly (cleaner provenance)
      - kai-memory (km-*): possibly small lift — kai journals occasionally
        reference users by slug

CHANGES vs iter1:
    Override `_make_capturing_handler` to run hits through
    `_recover_attributions(hits)` AFTER the legacy-title cleanup but
    BEFORE handing to library._format_hits. The original chunk_id and
    text are unchanged; only the metadata.subject_entity field is
    upgraded from 'unknown' to a recovered slug when one is identifiable.
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
    _clean_legacy_source_titles,
)
import library  # type: ignore[import-not-found]   # noqa: E402
from lib.corpus_vocab import KNOWN_USER_SLUGS  # type: ignore[import-not-found]   # noqa: E402
from lib.harness import ToolCallTrace  # type: ignore[import-not-found]   # noqa: E402

# ---------- attribution recovery ----------

# Real URI patterns observed in the canonical store:
#   legacy://saimemory/anna/anna-1467145ace13
#   legacy://saimemory/cf-cra-audit/persist-ed91edaa303b
#   /Users/samantha/.openclaw/workspace/sisters/sai-recovery/memory/2026-02-28-recovery-daily.md
_URI_SLUG_RE = re.compile(
    r"legacy://saimemory/(?P<slug>[a-zA-Z0-9][a-zA-Z0-9._-]*)/"
)
_TITLE_SLUG_RE = re.compile(
    r"saimemory/(?P<slug>[a-zA-Z0-9][a-zA-Z0-9._-]*)(?:\s*\(legacy\))?",
    re.IGNORECASE,
)
# Recognize known user slugs at start of text.
_TEXT_SCAN_PREFIX_CHARS = 200


def _normalize_slug(s: str) -> str:
    """Lowercase + ASCII fold. Mirror lib.corpus_vocab.normalize_user_slug
    behavior so 'Phil' → 'phil', 'MZea' → 'mzea'."""
    return s.strip().lower()


def _recover_subject_entity(hit: dict) -> tuple[str | None, str]:
    """Try to recover a canonical subject_entity for a hit whose current
    value is 'unknown'. Returns (recovered_value | None, source_signal).

    Recovery order (first match wins):
      1. URI pattern legacy://saimemory/<slug>/...
      2. source_title pattern "saimemory/<slug>" or "<slug> (legacy)"
      3. Known-slug name in the first 200 chars of text
    """
    md = hit.get("metadata") or {}

    # 1. URI pattern.
    src = md.get("source") if isinstance(md.get("source"), dict) else None
    uri = (src or {}).get("uri") or md.get("source_uri") or ""
    if uri:
        m = _URI_SLUG_RE.search(uri)
        if m:
            slug = _normalize_slug(m.group("slug"))
            if slug and slug not in {"_default_", "default"}:
                return _format_recovered(slug), "uri"

    # 2. source_title pattern.
    st = md.get("source_title") or ""
    m = _TITLE_SLUG_RE.search(st)
    if m:
        slug = _normalize_slug(m.group("slug"))
        if slug and slug not in {"_default_", "default", "ublib2", "saimemory"}:
            return _format_recovered(slug), "title"

    # 3. First-line text scan for known user slugs.
    text = (hit.get("text") or md.get("text") or "")[:_TEXT_SCAN_PREFIX_CHARS].lower()
    for slug in KNOWN_USER_SLUGS:
        # Word-boundary match against the slug as a name (slug is already
        # lowercase). Skip 'members' — it's a bulk profile container, not
        # a single person.
        if slug == "members":
            continue
        if re.search(rf"\b{re.escape(slug)}\b", text):
            return _format_recovered(slug), "text-scan"

    return None, ""


def _format_recovered(slug: str) -> str:
    """Map a recovered slug to its canonical subject_entity form.

    Per target-schema.md §1.3:
      - 'cf-*' → 'case:cf-<slug>'
      - 'sai-*' / 'kai' / 'aiko' / 'sean-callagy' → bare handle
      - everything else → 'user:<slug>'
    """
    s = slug.lower()
    if s.startswith("cf-"):
        return f"case:{s}"
    if s in {"kai", "aiko", "sean-callagy"} or s.startswith("sai-"):
        return s
    return f"user:{s}"


def _recover_attributions(hits: list[dict]) -> list[dict]:
    """For every hit with subject_entity=='unknown', try to recover.

    Returns a new list (or unchanged hits where no recovery was found).
    Annotates recovered hits with metadata['_recovered_subject_entity']
    + metadata['_recovery_source'] for the trace.
    """
    out: list[dict] = []
    for h in hits:
        md = h.get("metadata") or {}
        current = (md.get("subject_entity") or "").strip()
        if current and current != "unknown":
            out.append(h)
            continue
        recovered, source = _recover_subject_entity(h)
        if recovered is None:
            out.append(h)
            continue
        new_md = dict(md)
        new_md["subject_entity"] = recovered
        new_md["_recovered_subject_entity"] = recovered
        new_md["_recovery_source"] = source
        new_h = dict(h)
        new_h["metadata"] = new_md
        out.append(new_h)
    return out


# ---------- the candidate ----------


class Iter5AttributionRecoveryHarness(Iter1StripLegacyTitlesHarness):
    """Stacks subject_entity attribution recovery on top of iter1's title cleanup."""

    name: str = "iter5_attribution_recovery"

    def _make_capturing_handler(
        self,
        captured_calls: list[ToolCallTrace],
        captured_hits_flat: list[dict],
        captured_meta: dict[str, Any],
    ):
        """Override: clean titles AND recover attributions before format."""

        async def _handler(args: dict) -> str:
            t0 = time.time()
            turn_idx = sum(
                1 for c in captured_calls if c["name"] == "recall_context"
            ) + 1

            # --- mirror baseline validation block (verbatim) ---
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
            # iter1 step: clean placeholder titles.
            hits = _clean_legacy_source_titles(hits_raw)
            # iter5 step: recover attribution for unknown subjects.
            hits = _recover_attributions(hits)

            captured_hits_flat.extend(hits)
            captured_meta.setdefault("intents_used", []).append(intent)
            captured_meta["last_intent"] = intent
            captured_meta["last_payload"] = payload
            n_recovered = sum(
                1 for h in hits
                if (h.get("metadata") or {}).get("_recovered_subject_entity")
            )
            captured_meta["attributions_recovered"] = (
                captured_meta.get("attributions_recovered", 0) + n_recovered
            )

            formatted = library._format_hits(intent, query, hits)
            captured_calls.append(
                self._mk_call_trace(turn_idx, args, formatted, hits, duration_ms)
            )
            return formatted

        return _handler


AgentHarness = Iter5AttributionRecoveryHarness
