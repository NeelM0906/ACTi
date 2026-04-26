"""Library — the Unblinded knowledge corpus, exposed as a model-callable tool.

The corpus lives behind a hosted retrieval service (Railway-deployed
FastAPI over a Pinecone index). It contains the substrate that makes
Sohn coherent: Sean Callagy's body of work, the Unblinded Results
Formula, Zone Action and related teachings, dossiers on people in the
ecosystem, client cases, products, and identity records.

PUBLIC API (consumed by gateway.py via the Spark tool registry):

    RECALL_CONTEXT_TOOL    — OpenAI-shape tool schema for the model
    library_enabled()      — bool, true iff base URL + API key are set
    handle_recall_context(args) -> str
        Async tool handler for Spark. Calls the retrieval service,
        formats hits with [source_title] citations, returns a string
        the model can quote from. Never raises; on failure, returns
        a string starting with 'ERROR:'.

The credential lives only in this process's environment (loaded from
/etc/acti/library.env, gitignored). It is never handed to the model
or surfaced to clients.

DESIGN CHOICES

  - Eager tool advertisement (no system-prompt rules). The tool's
    description tells Sohn what's in the corpus and when to call.
    Same pattern as generate_image / load_skill.
  - Six intents, each with smart server-side defaults (top_k,
    namespace weighting, hybrid alpha, rerank, recency boost). The
    model picks intent based on question shape; intent's metadata
    is in the tool description so the model knows what each does.
  - subject_entity / case_id are optional in the schema but required
    server-side for some intents — we mirror that, and the format
    string explains the requirement so the model self-corrects.
  - Output formatting: numbered hits with `[source_title]` citations,
    so Sohn can quote naturally ("From the Unblinded Formula: ...").
"""
from __future__ import annotations

import os
import sys
from typing import Any

import httpx


# ---------- configuration ----------

LIBRARY_BASE_URL = os.environ.get("ACTI_LIBRARY_BASE_URL", "").rstrip("/")
LIBRARY_API_KEY = os.environ.get("ACTI_LIBRARY_API_KEY", "")
LIBRARY_TIMEOUT = float(os.environ.get("ACTI_LIBRARY_TIMEOUT", "30"))
LIBRARY_MAX_HITS = int(os.environ.get("ACTI_LIBRARY_MAX_HITS", "8"))


def _log(msg: str) -> None:
    print(msg, file=sys.stdout, flush=True)


def library_enabled() -> bool:
    return bool(LIBRARY_BASE_URL and LIBRARY_API_KEY)


# ---------- tool schema (registered with Spark) ----------

RECALL_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "recall_context",
        "description": (
            "Search the Unblinded knowledge corpus — Sean Callagy's body of work, "
            "the Unblinded Results Formula, Zone Action and related teachings, dossiers "
            "on people in the ecosystem (Sean, Adam, Kai, and others with `user:*` slugs), "
            "client cases (with `cf-*` slugs like cf-denver-family-law), products, and "
            "structural identity records. The corpus is ~50k chunks across teachings (36k), "
            "user content (8.7k), memory (1k), cases (61), products (171), and identity (7).\n\n"
            "CALL THIS WHENEVER you need to look up Unblinded-specific knowledge to ground "
            "your answer — concepts, methodology, named people, named cases, or anything "
            "that would otherwise require you to fabricate. Returns ranked text snippets "
            "with `[source_title]` citations you can quote from.\n\n"
            "Choose `intent` based on the question shape:\n"
            "  - `explain-teaching` — \"what is X / explain Y\" (Unblinded concepts, methodology)\n"
            "  - `person-recall` — \"tell me about person P / what's their story\" (REQUIRES `subject_entity`)\n"
            "  - `case-lookup` — per-case content (REQUIRES `case_id`)\n"
            "  - `kai-memory` — Kai's recent journal + cross-context\n"
            "  - `continuity-snapshot` — long structural identity dump (REQUIRES `subject_entity`)\n"
            "  - `general` — catch-all when none of the above clearly fits"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The natural-language question or topic to search for. Be specific; "
                        "include named entities and key concepts. The retrieval is hybrid "
                        "(semantic + BM25), so both meaning and exact terms help."
                    ),
                },
                "intent": {
                    "type": "string",
                    "enum": [
                        "explain-teaching",
                        "person-recall",
                        "case-lookup",
                        "kai-memory",
                        "continuity-snapshot",
                        "general",
                    ],
                    "description": "Retrieval strategy. Defaults to `general` if you're unsure.",
                    "default": "general",
                },
                "subject_entity": {
                    "type": "string",
                    "description": (
                        "Slug for the person this is about. Required for `person-recall` "
                        "and `continuity-snapshot`. Format: `user:<slug>` (e.g. "
                        "`user:adam-gugino`, `user:sean-callagy`)."
                    ),
                },
                "case_id": {
                    "type": "string",
                    "description": (
                        "Slug for the case. Required for `case-lookup`. Format: `cf-<slug>` "
                        "(e.g. `cf-denver-family-law`)."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Max number of hits to return. Optional — each intent has a sensible "
                        "default (5–80). Lower this for tighter results."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


# ---------- service client ----------

async def _call_retrieve(payload: dict) -> dict:
    """POST /v1/retrieve and return the parsed response. On error, returns
    {'error': '<message>'}.
    """
    headers = {
        "Authorization": f"Bearer {LIBRARY_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=LIBRARY_TIMEOUT) as c:
            resp = await c.post(
                f"{LIBRARY_BASE_URL}/v1/retrieve",
                json=payload,
                headers=headers,
            )
    except httpx.HTTPError as e:
        return {"error": f"library service unreachable: {e}"}
    if resp.status_code != 200:
        return {"error": f"library service {resp.status_code}: {resp.text[:300]}"}
    try:
        return resp.json()
    except ValueError as e:
        return {"error": f"library returned non-JSON: {e}"}


def _format_hits(intent: str, query: str, hits: list[dict]) -> str:
    """Format ranked hits into a string the model can quote from.

    Layout: numbered list with `[source_title]` citation prefix on each
    entry. Truncate per-snippet text at ~1200 chars — the corpus chunks
    are usually under that, but defensive against oversize text.
    """
    if not hits:
        return (
            f"No hits in the Unblinded library for query={query!r} intent={intent!r}. "
            f"Either the corpus genuinely lacks this content or the question phrasing "
            f"didn't match well. Either rephrase / try a different intent, or proceed "
            f"without the lookup and tell the user."
        )
    lines = [
        f"Library returned {len(hits)} hit(s) for query={query!r} intent={intent!r}.",
        "Use these snippets to ground your answer. Cite by source_title where natural.",
        "",
    ]
    for i, h in enumerate(hits, 1):
        m = h.get("metadata", {}) or {}
        src = m.get("source_title", "(unknown source)")
        ns = m.get("namespace", "?")
        subj = m.get("subject_entity")
        score = h.get("score", 0.0)
        text = (h.get("text") or "").strip()
        if len(text) > 1200:
            text = text[:1200].rstrip() + "…"
        header = f"{i}. [{src}] (ns={ns}"
        if subj and subj != "unknown":
            header += f", subject={subj}"
        header += f", score={score:.2f})"
        lines.append(header)
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------- public tool handler ----------

async def handle_recall_context(args: dict) -> str:
    """Spark tool handler for `recall_context`. Returns the formatted
    tool-result text the model will see.
    """
    if not library_enabled():
        return "ERROR: the Unblinded library is not configured on this deployment."

    query = (args.get("query") or "").strip()
    if not query:
        return "ERROR: `query` is required."

    intent = args.get("intent") or "general"

    # Server-side enforcement of required params per intent. Mirroring
    # the service's own validation gives the model a clearer error
    # rather than a 4xx from upstream.
    if intent in {"person-recall", "continuity-snapshot"} and not args.get("subject_entity"):
        return (
            f"ERROR: intent={intent!r} requires `subject_entity` "
            f"(format: `user:<slug>`, e.g. `user:adam-gugino`). "
            f"Either provide it, or call with intent=general."
        )
    if intent == "case-lookup" and not args.get("case_id"):
        return (
            f"ERROR: intent='case-lookup' requires `case_id` "
            f"(format: `cf-<slug>`, e.g. `cf-denver-family-law`). "
            f"Either provide it, or call with intent=general."
        )

    payload: dict[str, Any] = {"query": query, "intent": intent}
    if args.get("subject_entity"):
        payload["subject_entity"] = args["subject_entity"]
    if args.get("case_id"):
        payload["case_id"] = args["case_id"]
    if args.get("top_k"):
        payload["top_k"] = max(1, min(int(args["top_k"]), LIBRARY_MAX_HITS))
    else:
        payload["top_k"] = LIBRARY_MAX_HITS

    result = await _call_retrieve(payload)
    if "error" in result:
        return f"ERROR: {result['error']}"

    hits = result.get("hits") or []
    return _format_hits(intent, query, hits)
