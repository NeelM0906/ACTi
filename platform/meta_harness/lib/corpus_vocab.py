"""Frozen vocabularies for the ACTi-Unblinded corpus.

Sources of truth:
  - Subject entities + content types: target-schema.md §1.3 + §1.2 enum
  - Topic vocabulary:                 target-schema.md §2.4
  - Real per-namespace distributions: 2026-04-27 scan of
                                      ~/Desktop/ACTi_base/pinecone-audit/canonical/

Use these constants instead of inline literals so a single edit propagates.
"""
from __future__ import annotations

import re

# ---------- namespaces ----------

NAMESPACES: frozenset[str] = frozenset({
    "teachings",
    "memory",
    "users",
    "cases",
    "products",
    "identity",
})

# Real corpus sizes (2026-04-24 canonical store).
NAMESPACE_SIZES: dict[str, int] = {
    "teachings": 157_175,
    "users": 25_052,
    "memory": 3_829,
    "products": 786,
    "cases": 149,
    "identity": 35,
}


# ---------- content types ----------

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


# ---------- subject_entity vocabulary ----------

# Fixed handles (founder, AI agents, system).
FIXED_SUBJECT_ENTITIES: frozenset[str] = frozenset({
    "sean-callagy",
    "sai-prime",
    "sai-forge",
    "sai-scholar",
    "sai-recovery",
    "sai-seven-levers",
    "kai",   # legacy AI agent alias, retained for historical memory
    "aiko",  # legacy AI agent alias
    "system",
    "unknown",
})

# Slug grammar for users/cases. ASCII-only, lowercase letters/digits + . _ -
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_USER_RE = re.compile(r"^user:[a-z0-9][a-z0-9._-]*$")
_CASE_RE = re.compile(r"^case:cf-[a-z0-9][a-z0-9._-]*$")


def is_valid_subject_entity(s: str) -> bool:
    """True iff `s` is a canonical subject_entity per target-schema.md §1.3.

    Accepts:
      - any handle in FIXED_SUBJECT_ENTITIES
      - 'user:<slug>' where <slug> matches the slug grammar
      - 'case:cf-<slug>' similarly

    Note: production's recall_context tool description and the canonical
    grammar disagree on whether the founder is `sean-callagy` or
    `user:sean-callagy`. The canonical answer is `sean-callagy` (no
    `user:` prefix); production's tool example is wrong. See open question
    in domain_spec.md #11.
    """
    if not isinstance(s, str) or not s:
        return False
    if s in FIXED_SUBJECT_ENTITIES:
        return True
    return bool(_USER_RE.match(s) or _CASE_RE.match(s))


def normalize_user_slug(s: str) -> str:
    """Lowercase + ASCII-fold. Used by the attribution recovery pass.

    Example: 'Phil' → 'phil', 'MZea' → 'mzea', 'thomas-ferman' → 'thomas-ferman'.
    Does not add the 'user:' prefix — caller composes that.
    """
    return s.strip().lower()


# Real `users` namespace distribution (2026-04-27 scan).
# Useful for the proposer to know which user names actually have records,
# vs. the 24,624 "unknown" records that need attribution recovery.
KNOWN_USER_SLUGS: frozenset[str] = frozenset({
    "anna",
    "thomas-ferman",
    "ali",
    "phil",
    "mj",
    "mzea",
    "jared",
    "miko",
    "members",  # bulk profile records, not a single user
})

# Real `cases` namespace examples (substring-tested in scenario authoring).
KNOWN_CASE_SLUGS_SAMPLE: frozenset[str] = frozenset({
    "cf-cra-audit",
    "cf-cra-audit-toronto",
    "cf-cra-audit-gta",
    "cf-acti-legal-summit",
    "cf-la-criminal",
    "cf-denver-family-law",
    "cf-toronto-tax-cra",
})


# ---------- intents (the recall_context tool enum) ----------

INTENTS: frozenset[str] = frozenset({
    "explain-teaching",
    "person-recall",
    "case-lookup",
    "kai-memory",
    "continuity-snapshot",
    "general",
})

# Per-intent required params (mirrors library.py:handle_recall_context).
INTENT_REQUIRED_PARAMS: dict[str, frozenset[str]] = {
    "explain-teaching": frozenset(),
    "person-recall": frozenset({"subject_entity"}),
    "case-lookup": frozenset({"case_id"}),
    "kai-memory": frozenset(),
    "continuity-snapshot": frozenset({"subject_entity"}),
    "general": frozenset(),
}


# ---------- topic controlled vocabulary ----------

# Starter set per target-schema.md §2.4. Expansion via PR + linter.
TOPICS: frozenset[str] = frozenset({
    # Sean Callagy / Unblinded teaching frameworks
    "zone-action",
    "zone-of-genius",
    "heroic-unique-identity",
    "mastery-scale",
    "seven-levers",
    "pareto-39-elements",
    "sage-strategy",
    "barry-framework",
    "formula-cascade",
    "coaching-diagnostic",
    "relational-container",
    "achievement-acknowledgment",
    "recovery-transformation",
    "leverage-scale",
    # Business / operational
    "marketing",
    "sales-roleplay",
    "product-ops",
    "ip-report",
    "api-doc",
    "legal-practice",
    "case-strategy",
    "intake",
    "discovery",
    # Agent / memory
    "sister-kai",
    "sister-aiko",
    "sister-prime",
    "sister-forge",
    "sister-scholar",
    "sister-recovery",
    "sister-seven-levers",
    "daily-journal",
    "longterm-insight",
    "continuity-transfer",
    "worklog",
    "news-summary",
    # Administrative
    "identity",
    "mission",
    "protocol",
    "test-artifact",
})


# ---------- vendor-name forbidden list (mirrors rubric.yaml identity_lockdown) ----------

# Used only by the harness baseline's optional pre-filter. The eval rubric
# is the source of truth; this is a programmatic short-circuit candidates
# can opt into.
FORBIDDEN_VENDOR_TOKENS: frozenset[str] = frozenset({
    "gpt", "claude", "llama", "mistral", "gemini", "qwen",
    "deepseek", "kimi", "moonshot", "anthropic", "openai",
})
