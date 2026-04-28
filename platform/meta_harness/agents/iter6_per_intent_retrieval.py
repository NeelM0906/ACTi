"""iter6_per_intent_retrieval — case-lookup-specific retrieval policy.

PARENT: iter5_attribution_recovery (stacks both iter1 + iter5 changes)
HYPOTHESIS:
    The case-lookup intent (cl-*) is consistently the second-weakest
    intent across iter-3 (91.3), iter-4 (91.3) — only person-recall
    scores lower. The audit's run5 retrieval benchmark documented:

      - case-lookup with rerank: Hit@1 80%, p95 7735ms (slow + flat)
      - case-lookup without rerank: Hit@1 80%, p95 1358ms (fast + flat)

    Rerank doesn't help case-lookup but the production retrieval service
    has it on by default. We can't toggle rerank over HTTP (Q11
    constraint), but we CAN:
      (a) request a larger top_k to give the model more material, and
      (b) re-sort hits client-side to prioritize hits whose
          namespace=='cases' AND metadata.case_id matches the
          requested case_id (or whose chunk_id starts with
          'case-folder:cf-<requested_slug>').

    The case-lookup eval scenarios are all about ONE case at a time
    (cl-001 = CRA Toronto, cl-002 = Legal Summit, etc.). Putting the
    case-matched hits first lets the model open with the most relevant
    excerpts, lifts citation specificity, and avoids the model
    paraphrasing case A while citing chunks from case B.

    Predicted effect:
      - cl-* intent mean: 91.3 → ≥94
      - intent_and_param_correctness: stable (no schema change)
      - other intents: unchanged (we only intervene on case-lookup)

CHANGES vs iter5:
    1. When intent=='case-lookup', override top_k to max(top_k, 12).
    2. After hits return, sort case-namespace hits whose chunk_id matches
       the requested case_id BEFORE all other hits, preserving the
       original score-order within each group. Truncate back to the
       caller's intended top_k after re-sort.
"""
from __future__ import annotations

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
    _clean_legacy_source_titles,
)
from agents.iter5_attribution_recovery import (  # type: ignore[import-not-found]   # noqa: E402
    Iter5AttributionRecoveryHarness,
    _recover_attributions,
)
import library  # type: ignore[import-not-found]   # noqa: E402
from lib.harness import ToolCallTrace  # type: ignore[import-not-found]   # noqa: E402

# ---------- per-intent overrides ----------

CASE_LOOKUP_TOP_K = 12  # bumped from default 8


def _case_id_matches(hit: dict, case_id: str) -> bool:
    """True iff this hit looks like it's about the requested case.

    Matches by chunk_id prefix (`case-folder:<case_id>:`) OR by
    metadata.subject_entity ending in case_id. Case-insensitive on slugs.
    """
    md = hit.get("metadata") or {}
    cid = (case_id or "").strip().lstrip("/")
    if not cid:
        return False
    cid_l = cid.lower()
    chunk_id = (hit.get("id") or md.get("chunk_id") or "").lower()
    if chunk_id.startswith(f"case-folder:{cid_l}:"):
        return True
    subj = (md.get("subject_entity") or "").lower()
    # subject_entity for cases is 'case:cf-<slug>'.
    if subj == f"case:{cid_l}" or subj.endswith(f"case:{cid_l}"):
        return True
    return False


def _prioritize_case_hits(
    hits: list[dict], case_id: str | None, target_top_k: int
) -> list[dict]:
    """Re-sort hits to put case_id-matching hits first.

    Preserves original within-group order (which is score-order). Truncates
    to `target_top_k` AFTER re-sort, so a case-matching hit ranked 11
    overall still gets surfaced if target_top_k=8.
    """
    if not case_id:
        return hits[:target_top_k]
    matching: list[dict] = []
    other: list[dict] = []
    for h in hits:
        if _case_id_matches(h, case_id):
            matching.append(h)
        else:
            other.append(h)
    return (matching + other)[:target_top_k]


# ---------- the candidate ----------


class Iter6PerIntentRetrievalHarness(Iter5AttributionRecoveryHarness):
    """Stacks per-intent retrieval policy on top of iter5 + iter1."""

    name: str = "iter6_per_intent_retrieval"

    def _make_capturing_handler(
        self,
        captured_calls: list[ToolCallTrace],
        captured_hits_flat: list[dict],
        captured_meta: dict[str, Any],
    ):
        """Override: case-lookup gets top_k bump + case-prioritized sort."""

        async def _handler(args: dict) -> str:
            t0 = time.time()
            turn_idx = sum(
                1 for c in captured_calls if c["name"] == "recall_context"
            ) + 1

            # --- baseline validation (verbatim) ---
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

            # --- iter6: bump top_k for case-lookup ---
            requested_top_k = (
                int(args["top_k"]) if args.get("top_k") else library.LIBRARY_MAX_HITS
            )
            requested_top_k = max(1, min(requested_top_k, library.LIBRARY_MAX_HITS))
            if intent == "case-lookup":
                fetch_top_k = max(requested_top_k, CASE_LOOKUP_TOP_K)
            else:
                fetch_top_k = requested_top_k
            payload["top_k"] = fetch_top_k

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
            # iter5 step: recover attribution.
            hits = _recover_attributions(hits)
            # iter6 step: case-lookup re-prioritization.
            n_prioritized = 0
            if intent == "case-lookup":
                before_ids = [h.get("id") for h in hits[:requested_top_k]]
                hits = _prioritize_case_hits(
                    hits, args.get("case_id"), requested_top_k
                )
                after_ids = [h.get("id") for h in hits[:requested_top_k]]
                n_prioritized = sum(
                    1 for i, hid in enumerate(after_ids)
                    if i < len(before_ids) and hid != before_ids[i]
                )

            captured_hits_flat.extend(hits)
            captured_meta.setdefault("intents_used", []).append(intent)
            captured_meta["last_intent"] = intent
            captured_meta["last_payload"] = payload
            captured_meta["case_lookup_reranks"] = (
                captured_meta.get("case_lookup_reranks", 0) + n_prioritized
            )

            formatted = library._format_hits(intent, query, hits)
            captured_calls.append(
                self._mk_call_trace(turn_idx, args, formatted, hits, duration_ms)
            )
            return formatted

        return _handler


AgentHarness = Iter6PerIntentRetrievalHarness
