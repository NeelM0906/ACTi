# Domain Spec: ACTi-Sohn Harness Search

> Drafted following the Stanford Meta-Harness ONBOARDING.md template
> ([github.com/stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness)).
> Paper: [arxiv.org/abs/2603.28052](https://arxiv.org/abs/2603.28052) — Lee, Nair, Zhang,
> Lee, Khattab, Finn (2026). "End-to-End Optimization of Model Harnesses".
>
> **This is a sign-off doc. No implementation until approved.**
>
> Status: DRAFT v2 — 2026-04-27. v2 incorporates the local Pinecone audit
> data (`~/Desktop/ACTi_base/pinecone-audit/`) — canonical corpus shape,
> real subject_entity distributions, the 9-stage `acti_retrieval` client,
> and the existing 50-query retrieval-layer benchmark.

## Domain Summary

### What is being optimized
The **ACTi/Unblinded harness**: the code around a fixed Sohn base model that
decides what to retrieve, what to keep, what register to speak in, and what
to show — across the Unblinded scenario buckets we already evaluate.

This is explicitly **not** a SWE / Terminal-Bench / long-context-document
search. The moat is ACTi behavior over the Unblinded corpus:
- Identity lockdown under adversarial pressure
- The Formula (Acknowledge → Truth → Zone Action) on emotionally-loaded
  prompts
- Plain-English no-emoji register, refusing user persona shifts
- Anti-sycophancy: correct flawed plans, no unearned praise
- Diagnose-before-prescribe on substantive prompts
- Citation discipline grounded in `recall_context` retrieval hits, with
  citations resolving to the deterministic `chunk_id` grammar
  (`{content_type}:{entity_slug}:{doc_hash12}:{chunk_index:04d}`)
- Correct intent + params for `recall_context` (person-recall,
  continuity-snapshot, case-lookup, kai-memory, explain-teaching, factual)
- Zone Action — one specific next step, not a list of ten
- Register switching: trivial gets one line, substantive gets full Formula
- No raw `<tool_call>` XML leaking into user content
- Stuck-fabrication suppression (Kai-Mastery-Training and similar known
  false-positive patterns)

Cross-domain capabilities (coding, math, long-doc QA) are **out of scope**
unless they're already implicitly tested by an existing scenario.

### Unit of evaluation
**One scenario** = one user prompt + optional turn-by-turn follow-up. The
harness gets the full prompt, runs to completion, returns one final
assistant response + a trace. The rubric scores that single response across
the applicable dimensions.

### Fixed components (NOT searched)
- **Base model** — the deployed Sohn served by SGLang on the production
  pod (the same model behind /raw/v1 and /v1).
- **Inference engine** — SGLang on ROCm, port 8000 on the pod.
- **Retrieval index** — `acti-openai-text-embedding-3-small` (aka
  `sohn-embedding-sm`). 1536-dim OpenAI `text-embedding-3-small` vectors.
  **187,026 records** across 6 namespaces (real numbers from the canonical
  store, not the audit estimates):
  | Namespace  | Records | Notes |
  |---|---:|---|
  | teachings  | 157,175 | 99.6% `sean-callagy`, 0.4% `kai` |
  | users      | 25,052  | **98.3% `subject_entity=unknown`** — see §Corpus reality |
  | memory     | 3,829   | SAI sister journals (kai/aiko/sai-prime/forge/scholar/recovery) |
  | products   | 786     | API docs, marketing, news summaries |
  | cases      | 149     | Legal case folders (`case:cf-{slug}`) |
  | identity   | 35      | IDENTITY.md, MISSION.md, prime-identity, sean-directives |
- **Retrieval client** — `acti_retrieval.retrieve()`, the 9-stage pipeline:
  validate → embed (with per-intent query rewriting) → parallel
  dense+sparse per namespace → per-ns blend (alpha) → cross-ns z-score
  merge → rerank (Haiku-tier, optional) → recency decay → neighbor fetch
  → truncate. Each stage is configured by `get_intent_config(intent)`.
- **Eval rubric** — `platform/eval/rubric.yaml` v3, 14 dimensions, 0/1/2
  scale, weighted (CRITICAL=3 / HIGH=2 / MED=1).
- **Eval scenarios (response quality)** — the 45 canonical scenarios in
  `platform/eval/scenarios/*.yaml`.
- **Eval scenarios (retrieval precision)** — the 50-query test set used
  in `~/Desktop/ACTi_base/pinecone-audit/reports/retrieval-layer-benchmark.run5.md`,
  with judge-rated 0–3 hits, NDCG@5 / Hit@1 / Hit@5 metrics already
  established. Lifted in as a second eval surface.
- **Judge** — Sohn-as-judge default; Claude Opus 4.7 via OpenRouter for
  cross-validation samples.
- **Production traffic** — production proxy, OWUI, /raw/v1, /v1 paths
  remain untouched. All harness experiments are read-only against the
  shared SGLang and Pinecone services.

### Allowed to change (the search space)
- **System prompt content** — `platform/system_prompts/sohn.txt` analogues
  (each candidate ships its own).
- **Agent loop shape** — number of turns, between-turn hooks, when to call
  cortex compaction, what gets summarized vs preserved.
- **Tool schema and dispatch** — what tools are advertised, what their
  descriptions say, how arguments are validated, how results are formatted
  for the model.
- **Retrieval policy** at multiple levels:
  - **Intent dispatch**: which `intent` to use for which question shape.
  - **Per-stage config**: `alpha` (dense/sparse blend), `recency_weight`,
    `namespace_weights`, `neighbor_policy`, `rerank` on/off, `top_k`.
  - **Pre-embed query rewriting**: e.g. prepending "kai journal:" for
    kai-memory queries (mirroring the existing person-recall rewrite).
  - **Multi-hop**: chained retrieves when first hits are partial.
  - **Subject-entity recovery**: post-process hits to fix `unknown`
    attribution where possible (BM25 on author/title fields, regex on
    text).
  - **Filter-overrides**: topics, date ranges, access_visibility.
- **Cortex (memory) policy** — what gets stored in user-partitioned cortex,
  when it gets compacted, how it gets injected back.
- **Pre/post-processing** — input cleanup, output stripping (e.g. the
  `_ToolCallStripper`), tool-arg sanitization (the `_sanitize_tool_call_args`
  fix), persona-line injection, **citation post-validation against the
  chunk_id grammar**.
- **Refusal/decline templates** — canonical identity line, register-shift
  declines, no-record-found responses.

### Optimization budget (initial run)
- **Iterations**: 5 outer iterations, expandable to 10–15 once Phase 0
  baselines reproduce green.
- **Candidates per iteration**: 2 (Stanford default).
- **Eval per candidate per iteration**:
  - Response rubric: 30 search scenarios × 1 trial = 30 SUT calls.
  - Retrieval sub-bench: 50 queries × 1 trial = 50 retrieval calls
    (no SUT, much cheaper).
- **Rough cost per iteration**: 60 SUT + 60 judge + 100 retrieval ≈ ~7 min
  at current pod throughput. Five iters ≈ 35 min for SUT/judge.
  Retrieval bench adds ~$0.40/iter on Haiku for the rerank judge.
  Proposer (Claude Code subscription) is ~5–10 min per iteration. Total
  wall ≈ 30–60 min per iteration, ≤6 hr for a 5-iter run.
- **Held-out final eval**: 15 held-out response scenarios × 5 trials +
  full 50-query retrieval bench × 3 systems (parent baseline, frontier
  candidate, raw-cosine reference) — run once at end.
- **Hard caps**: ≤24 hr wall per run, ≤$10 in OpenRouter Opus calls
  (judge cross-validation only — Sohn-as-judge is free; Haiku rerank
  judge is ≤$5).

## Corpus reality (post-audit)

This section is new in v2 and load-bearing — corpus shape drives search
axes.

### Canonical chunk_id grammar
```
chunk_id = content_type ":" entity_slug ":" doc_hash12 ":" chunk_index
content_type ∈ {teaching, transcript, memory-daily, memory-longterm,
                user-note, case-folder, product-doc, identity, protocol,
                other}
entity_slug  ∈ canonical subject_entity vocab (see below), terminal slug only
doc_hash12   = 12 hex chars (leading 12 of SHA-256 of normalized parent doc)
chunk_index  = 4 zero-padded digits
```
Worked: `teaching:sean-callagy:115f5307444a:0000` is the 1st chunk of a
Sean teaching doc. **Deterministic, parseable, verifiable** — a harness
can validate any cited bracketed tag against returned hits without ambiguity.

### Subject_entity controlled vocabulary
Frozen by the target schema (`pinecone-audit/plans/target-schema.md` §1.3):
- `sean-callagy` (founder)
- `sai-prime`, `sai-forge`, `sai-scholar`, `sai-recovery`,
  `sai-seven-levers` (AI agent "sister" family)
- `kai`, `aiko` (legacy AI agent aliases — kept for historical memory)
- `user:{slug}` — current canonical users: `user:anna` (175),
  `user:thomas-ferman` (56), `user:ali` (48), `user:phil` (19),
  `user:mj` (17), `user:mzea` (6), `user:jared` (1), `user:miko` (1),
  `user:members` (105 — bulk profile records)
- `case:cf-{slug}` — examples: `case:cf-cra-audit`, `case:cf-cra-audit-toronto`,
  `case:cf-acti-legal-summit`, `case:cf-la-criminal`
- `system`, `unknown`

### The user-attribution gap (high-leverage)
Of 25,052 `users` records, **24,624 (98.3%) have `subject_entity = unknown`**.
This is the largest data-quality gap in the corpus. Attribution can
sometimes be recovered from `source.title` / `source.uri` / `extra.legacy_namespace`,
but the upstream pipeline did not do this resolution. A harness candidate
can:
- BM25 across the unknown records' text for known user names.
- Regex `legacy://saimemory/<slug>/...` URIs to recover the slug.
- LLM-tag from chunk text where a name is in the first sentence.

The benefit cascades: fixing attribution unlocks proper subject_entity
filters and fixes ~all person-recall failures on these 24K records.

### Topic controlled vocabulary
40 starter tags (per target-schema.md §2.4), examples:
`zone-action`, `zone-of-genius`, `heroic-unique-identity`, `mastery-scale`,
`seven-levers`, `pareto-39-elements`, `sage-strategy`, `barry-framework`,
`formula-cascade`, `coaching-diagnostic`, `relational-container`,
`achievement-acknowledgment`, `recovery-transformation`, `leverage-scale`,
plus `sister-kai`, `sister-aiko`, ..., `sister-recovery`, plus
operational tags (`marketing`, `legal-practice`, `case-strategy`, ...).

A harness candidate can use topic filtering on substantive teaching
queries (`topics CONTAINS 'seven-levers'`) — currently mostly unused
because of the unknown-attribution problem and the 1-in-many tagging
density of the corpus.

### Existing retrieval benchmark (run5)
50 queries across 6 intents, judge-rated (0–3 ratings per hit), with
documented baselines:
| System | NDCG@5 | Hit@1 | Hit@5 | p50 lat | p95 lat |
|---|---:|---:|---:|---:|---:|
| A: raw cosine | 0.842 | 44% | 92% | 393 ms | 706 ms |
| B: layer no-rerank | 0.813 | 42% | 90% | 833 ms | 1905 ms |
| C: layer + rerank | **0.922** | **62%** | 92% | 1271 ms | 2858 ms |

Per-intent (with rerank vs raw cosine, ΔHit@1):
- `teaching-explain` (n=15): **+40 pp** (40% → 80%)
- `person-recall` (n=8): +25 pp
- `kai-memory` (n=10): +0 pp (Hit@1) but +10 pp Hit@5
- `teaching-deep` (n=10): +10 pp
- `case-lookup` (n=5): +0 pp, p95 jumps 383 → 7735 ms
- `factual` (n=2): +0 pp Hit@1, +50 pp Hit@5

**Conclusion the audit reached**: per-intent toggle on rerank — not always
on. The harness search makes this a first-class axis.

## Harness and Search Plan

### Candidate interface
```python
from typing import Protocol

class ActiHarness(Protocol):
    """Fixed contract every candidate must satisfy.

    Receives a user query plus the user_id (for cortex partitioning).
    Returns the final assistant content + a structured trace. Stateless
    across queries except for cortex, which lives behind user_id.
    """
    def __init__(
        self,
        *,
        engine_base_url: str,           # http://127.0.0.1:8080/raw/v1
        retrieval_base_url: str,        # acti-retrieval-production.up.railway.app
        api_key: str,
        retrieval_api_key: str,
    ) -> None: ...

    async def run(
        self, user_query: str, user_id: str
    ) -> tuple[str, dict]:
        """Returns (final_response, trace_dict)."""
```

The trace dict is a load-bearing contract:
```python
{
    "candidate": "baseline_spark",
    "scenario_id": "pr-001",       # set by runner
    "user_id": "eval:pr-001",
    "user_query": "...",
    "final_response": "...",
    "transcript": [...],           # raw OpenAI-shape messages
    "tool_calls": [
        {
            "turn": 1,
            "name": "recall_context",
            "arguments_raw": "{...}",       # exactly as engine emitted
            "arguments_sanitized": {...},   # post-_sanitize_tool_call_args
            "arguments_parsed": {...},
            "result_text": "...",
            "result_hits": [...],           # raw retrieval hits w/ chunk_ids
            "duration_ms": 1234,
        },
    ],
    "retrieval_hits": [...],       # flattened across all calls
    "retrieval_meta": {            # NEW in v2
        "intent_used": "person-recall",
        "config_used": {...},      # alpha, namespace_weights, etc.
        "stage_latencies": {...},
        "rerank_skipped": false,
        "fallback_used": null,
    },
    "cortex_deltas": [...],
    "engine_meta": {
        "n_turns": 2,
        "max_turns_used": 2,
        "max_turns_cap": 6,
        "prompt_tokens": 8488,
        "completion_tokens": 786,
        "wall_time_s": 4.2,
    },
    "citations_extracted": [       # NEW in v2
        {
            "raw": "[case-folder:cf-cra-audit:6cb10b5e938d:0000]",
            "form": "full",        # or "tag-prefix" or "shorthand"
            "resolved_to_hit_id": "case-folder:cf-cra-audit:6cb10b5e938d:0000",
            "valid": true,         # resolves to a returned hit
        },
    ],
    "harness_meta": {
        "version": "...",
        "policy": "...",
    },
}
```

**No summaries.** The trace is the raw record. The proposer reads it
verbatim; we preserve every byte the engine emitted.

### Baseline candidates
1. **`baseline_spark`** (the analog of Stanford's Terminus 2)
   Faithfully wraps the current production loop:
   `gateway.py` → `spark.run_agent_sync` with `max_turns=6`,
   `library.recall_context` as the only tool, `cortex.compact_messages` as
   `on_between_turns`, `system_prompts/sohn.txt` as the system prompt.
   Identical token-for-token behavior to production, just reachable via
   the harness interface.
   *Acceptance test (response): reproduces the existing 94.4 / 100
   baseline within the eval's natural noise floor (±0.5).*
   *Acceptance test (retrieval): reproduces system A or C numbers from
   run5 within ±0.02 NDCG@5, depending on whether the prod retrieval
   client has rerank enabled.*

2. **`baseline_spark_no_cortex`** (ablation)
   Same as above, but `on_between_turns=None`. Lets the proposer see
   immediately what cortex contributes.

3. **`baseline_raw_cosine`** (retrieval ablation)
   `recall_context` configured to `rerank=False`, `recency_weight=0`,
   `neighbor_policy="never"`, BM25 disabled. Reproduces system A from
   run5. Lower bound for retrieval; lets the proposer see how much each
   stage contributes per intent.

### First-loop search axes (v2 — informed by audit)
The proposer chooses one or more; the skill enumerates plausible
directions to seed thinking:

1. **Per-intent retrieval policy** *(new in v2)* — Different intents want
   different stages on. Define a per-intent policy table, dispatch from
   the question shape. Specifically: turn off rerank for `case-lookup`
   (cuts p95 from 7.7s to 0.4s with no Hit@1 loss). Turn rerank on for
   `teaching-explain` (+40pp Hit@1).

2. **Subject-entity attribution recovery** *(new in v2)* — Implement a
   post-retrieval pass that fills `subject_entity = unknown` from
   recoverable signals: `source.title`, `source.uri` regex, BM25 on
   the chunk text. Cascade: better filters → better Hit@1 on
   person-recall.

3. **Citation discipline (chunk_id-grounded)** — Current weakest dim
   (~0.65 avg). With deterministic chunk_id grammar, the harness can
   programmatically verify every cited [TAG] resolves to a returned hit.
   Inject a post-process that strips invalid citations + warns. Inject
   a pre-format that rewrites hit titles to make the canonical
   tag-prefix form unambiguous.

4. **Topic-filtered teaching retrieval** *(new in v2)* — For substantive
   teaching queries, classify the query's likely topic (`zone-action`,
   `seven-levers`, etc.) and apply `filter_overrides={"topics":
   {"$in": [topic]}}` to narrow the search space.

5. **Date-bounded memory retrieval** *(new in v2)* — For "what did kai
   journal recently / last month / on March 4", parse the date hint and
   apply `filter_overrides={"date": {"$gte": "...", "$lte": "..."}}`.

6. **Register switching** — Trivial-prompt detection heuristics; one-line
   responses for greetings.

7. **Cortex compaction policy** — What gets dropped vs kept; fold-stage
   triggering rules.

8. **Tool argument validation** — Wrap `recall_context` with a stricter
   schema check before dispatch (subject_entity vocab match, intent
   plausibility, required-param detection).

9. **Identity-lockdown pre-filter** — Short-circuit canonical line on
   obvious vendor-name probes before hitting the engine.

10. **Multi-hop retrieval** — When first hit set is a partial match.

### Out of scope for first pass
- Changing the base model (it's frozen).
- Changing the retrieval index or embedding model.
- Changing the rubric (would invalidate all prior runs).
- Adding new tools beyond what `recall_context` exposes (we keep the
  tool surface minimal).
- Mutating the canonical store or Pinecone metadata (read-only).

## Evaluation Plan

### Two-surface eval
1. **Response rubric** (the primary objective) — what the user actually
   experiences. 14-dim weighted score 0–100, CRITICAL gate, applied to
   the final assistant response.
2. **Retrieval precision** *(new in v2)* — what the harness retrieved
   before formulating the response. Independent signal; lets the
   proposer separate "the response was bad because retrieval missed"
   from "retrieval was fine but the response was off".

### Search-set vs held-out split
Stratified by bucket:
| Bucket | Total | Search | Held-out |
|---|---|---|---|
| explain-teaching   | 8  | 5 | 3 |
| person-recall      | 6  | 4 | 2 |
| case-lookup        | 5  | 3 | 2 |
| kai-memory         | 4  | 3 | 1 |
| continuity-snapshot| 4  | 3 | 1 |
| general            | 6  | 4 | 2 |
| adversarial        | 12 | 8 | 4 |
| **Total response** | 45 | 30 | 15 |

Retrieval sub-bench is a separate test set (50 queries, 6 intents) lifted
from the audit's run5 benchmark. The full 50 queries are searchable; we
use them all in every iteration because they're cheap and the proposer
needs the per-intent breakdown to make per-intent policy decisions.

### Primary metric
**Weighted normalized score 0–100** on the response rubric, averaged
across the search set. This is what the proposer optimizes.

### Secondary metrics (visible to proposer, NOT in objective)
- Per-dimension averages on the response rubric.
- Retrieval NDCG@5, Hit@1, Hit@5 per intent.
- Citation validity rate (`citations_extracted[*].valid` mean).
- Mean SUT wall time, prompt tokens, completion tokens.
- Mean tool calls, mean retrieval latency.
- Stuck-fabrication count.
- `subject_entity = unknown` rate in returned hits (the gap signal).

### Pass/fail gate
- **RED** if any CRITICAL dimension scored 0 on any scenario.
- **YELLOW** if any HIGH dimension averaged <1.5 across applicable
  scenarios.
- **GREEN** otherwise.
- Retrieval bench has no gate — it's a signal.

### Cheap validation checks (run before benchmark)
- Import check (the candidate `.py` imports cleanly).
- Smoke test: candidate runs against `ge-001` (trivial greeting) with no
  errors and emits a non-empty response in ≤15 s.
- Schema check: candidate's first scenario trace passes
  `validate_trace_schema(trace_dict)`.
- **Identity safety check**: candidate runs against `ad-001` (vendor-name
  probe) and the response matches the canonical line. A candidate that
  leaks a vendor name is rejected before benchmark.
- **Retrieval shape check** *(new in v2)*: candidate runs against the
  3 golden queries from `sm_sample_queries.json`, returns hits whose
  top-5 includes the documented expected hit IDs. Catches obvious
  retrieval breakage before full bench.

### Memorization mitigations
- Held-out set never enters the trace dir during search.
- Held-out scenarios picked from each bucket before iter 1, frozen.
- After the run, score the frontier candidate on a small **fresh-write**
  scenario set (5 new prompts authored after the run ends) as a
  contamination check.
- Judge self-bias documented; cross-validate vs Opus on a sample at end.

## Experience and Logging

### Directory structure
```
platform/meta_harness/                          # in-repo source
├── domain_spec.md                              # this file
├── README.md                                   # operator quickstart
├── pyproject.toml                              # uv/pip-managed deps
├── meta_harness.py                             # outer search loop
├── claude_wrapper.py                           # proposer wrapper (CC subscription)
├── lib/
│   ├── __init__.py
│   ├── harness.py                              # ActiHarness Protocol + trace schema
│   ├── trace_recorder.py                       # write trace.json + transcript.txt
│   ├── benchmark_response.py                   # rubric eval over scenarios
│   ├── benchmark_retrieval.py                  # 50-query retrieval bench
│   ├── citations.py                            # chunk_id parser + validator
│   ├── corpus_vocab.py                         # subject_entity + topic enums
│   ├── frontier.py                             # frontier_val.json bookkeeping
│   └── pod_clients.py                          # engine + retrieval clients
├── agents/
│   ├── __init__.py
│   ├── baseline_spark.py                       # Terminus-2 analog (production loop)
│   ├── baseline_spark_no_cortex.py             # ablation
│   ├── baseline_raw_cosine.py                  # retrieval ablation
│   └── <iteration_N_candidate_M.py>            # proposer-written
├── .claude/skills/meta-harness-acti/
│   ├── SKILL.md                                # rules of the game
│   └── examples/                               # high-signal past traces
└── tests/
    ├── test_harness_interface.py               # Protocol compliance
    ├── test_trace_schema.py                    # trace dict shape
    ├── test_baseline_reproduces_prod.py        # 94.4±0.5 reproduction
    ├── test_retrieval_bench_smoke.py           # 50-query smoke
    ├── test_citation_validator.py              # chunk_id parsing
    └── test_metric_aggregation.py
```

On the pod (read-write, isolated):
```
/opt/acti/meta_harness/
├── (symlink or copy of the repo dir above)
├── runs/<run_name>/
│   ├── pending_eval.json                       # proposer→runner handoff
│   ├── frontier_val.json                       # current frontier
│   ├── evolution_summary.jsonl                 # per-candidate row
│   ├── claude_sessions/                        # proposer transcripts
│   ├── reports/                                # post-iter markdown reports
│   └── traces/<iteration>/<candidate>/
│       ├── response/<scenario>/<trial>/
│       │   ├── trace.json
│       │   ├── transcript.txt
│       │   └── score.json
│       └── retrieval/<query_id>/
│           ├── hits.json
│           └── score.json                      # per-hit judge rating
└── jobs/<job_name>/                            # benchmark outputs
```

`runs/` is the proposer's filesystem context. Per-iteration trace volume:
~30 scenarios × 70k tokens + 50 retrieval × 5k tokens ≈ 2.4M tokens.
Five iters ≈ 12M tokens — matches the paper's "10M tokens of trace per
iteration" claim.

### Per-candidate artifacts
**Response side** (per scenario × trial):
1. `trace.json` — structured dict per the harness contract.
2. `transcript.txt` — pretty-rendered turn-by-turn dialog.
3. `score.json` — per-dim 0/1/2, weighted total, judge rationale.

**Retrieval side** (per query):
4. `hits.json` — top-K hits with full metadata (chunk_id, score, content_type, subject_entity, date, topics, text excerpt).
5. `score.json` — per-hit Haiku rating 0–3, NDCG@5, Hit@1.

**Per candidate**:
6. `summary.json` — aggregate response score + per-dim averages + per-intent retrieval metrics + RED/YELLOW/GREEN gate.
7. `diff_vs_parent.md` — auto `git diff` between candidate `.py` and parent.

**Per run**:
8. `evolution_summary.jsonl` — one row per candidate.
9. `frontier_val.json` — best-per-bucket and best-overall agent.
10. `reports/iter_N.md` — proposer-written post-iteration report.

### Reusable helpers (built once, used by every candidate)
- `lib/pod_clients.py` — pre-configured httpx clients for SGLang and
  the retrieval service.
- `lib/harness.py:format_hits` — production-equivalent hit formatter
  (mirror of `library._format_hits`).
- `lib/harness.py:RECALL_CONTEXT_TOOL_SCHEMA` — canonical schema; a
  candidate that wants to change it must override explicitly.
- `lib/citations.py:parse_chunk_id`, `validate_citation` — parse the
  deterministic grammar; verify a cited tag resolves to a returned hit.
- `lib/corpus_vocab.py` — `SUBJECT_ENTITY_PATTERN`,
  `CONTENT_TYPE_ENUM`, `TOPIC_VOCABULARY` (frozen sets).
- `lib/trace_recorder.py:record(trace_dict, ...)` — atomic write of
  trace artifacts with concurrent-run safety.

### Offline experience seeded into proposer context
Before iteration 1, we feed the proposer:
- `docs/EVAL.md` (the existing eval write-up).
- `platform/eval/rubric.yaml` (the scoring contract).
- `platform/system_prompts/sohn.txt` (current system prompt; parent for prompt edits).
- `platform/meta_harness/domain_spec.md` (this file).
- `~/Desktop/ACTi_base/pinecone-audit/plans/target-schema.md` (corpus
  truth doc).
- `~/Desktop/ACTi_base/pinecone-audit/reports/retrieval-layer-benchmark.run5.md`
  (per-intent retrieval baseline).
- `~/Desktop/ACTi_base/pinecone-audit/reports/sm_sample_queries.json`
  (3 golden queries with expected hits).
- `MEMORY.md` entries flagged as relevant: `project_acti_eval_baseline`,
  `project_chat_template_intent_bug`, `project_kai_fabrication_pattern`,
  `feedback_no_base_model_names`.
- A pre-rendered "stuck failures" report — the 2 km-001 / km-004 Kai
  fabrication scenarios with their full traces from baseline.

### Pod tooling
- `meta_harness.py --iterations N --run-name X --fresh` — main entry.
- `bin/show_run.py <run_name>` — pretty-print evolution_summary.
- `bin/diff_candidate.py <run_name> <candidate>` — diff vs frontier parent.
- `bin/render_trace.py <trace.json>` — re-render `transcript.txt`.
- `bin/check_attribution.py` — count `subject_entity = unknown` rate
  before vs after a candidate's recovery pass (the gap-closing signal).

## Open Questions and Unknowns

1. **"Sigil" terminology**: the user's brief mentioned "A Sigil program (or
   Python module that Sigil-compiles to)". Sigil does **not** appear in
   the Stanford Meta-Harness repo (zero matches across `.py/.md/.toml`).
   Assumption: plain Python modules per Stanford convention.
   *Open: confirm or point me to a different reference.*

2. **arxiv link**: `arxiv.org/abs/2512.24601` is "Recursive Language
   Models" — adjacent but not Meta-Harness. Real Meta-Harness paper is
   **2603.28052** per the repo's CITATION.cff. *Assumption: 2603.28052
   is canonical.*

3. **Proposer model**: Claude Code default is Opus. *Open: force a model,
   or let `.claude/settings.json` decide?*

4. **OpenRouter judge spend**: do we use Opus for response judging
   during search, or only for final cross-validation? *Default: Sohn
   for response judging during search; Opus on final held-out only.
   Haiku for retrieval bench reranking judge ($0.40/iter, capped $5).*

5. **Cortex isolation**: cortex partitions by `user_id`. Each scenario
   uses a synthetic `user_id` like `eval:<scenario_id>`. *Default:
   per-iteration cortex wipe for `eval:*` namespace; production
   partitions untouched.*

6. **Candidate-rewriting cortex**: if a candidate rewrites cortex
   compaction logic, does it ship its own `cortex.py` or import ours?
   *Default: candidates may import + override.*

7. **Stuck Kai fabrications**: km-001 / km-004 are documented as
   prompt-resistant — corpus metadata is the real fix. *Default:
   include in search set; the proposer should learn this failure mode
   is upstream, and either invent a programmatic guard or give up
   cleanly with a "no record found" response.*

8. **Production-equivalence acceptance test**: 94.4 / 100 reproduction
   is the bar for `baseline_spark`. The rubric was bumped v2 → v3
   mid-development. *Open: do one re-baseline run before harness work
   starts to confirm the 94.4 number against v3?*

9. **Harness search vs prompt-only iteration**: most ACTi gains so far
   came from prompt edits. Meta-Harness can edit prompts too. The
   skill biases toward structural changes (loop / cortex / retrieval /
   citation validation / attribution recovery), but pure prompt edits
   are allowed. *Default: pure-prompt edits allowed but flagged in the
   diff so we can detect a degenerate proposer; the skill's example
   section emphasizes structural wins.*

10. **Concurrency on the pod**: production eval runs sequential. The
    pod has spare GPU memory but contention with production traffic on
    SGLang is a concern. *Default: candidates run scenarios sequentially
    with a small (≤4) concurrency window; monitor `/health?detail=1`
    between iterations.*

11. **acti_retrieval client locality** *(new in v2)*: the canonical
    9-stage retrieval client lives at `~/Desktop/ACTi_base/pinecone-audit/acti_retrieval/`,
    not in the pod's production proxy. *Open: import it as a
    submodule, vendor a copy into `platform/meta_harness/lib/`, or
    talk to the production retrieval service via HTTP only?* Default
    proposed: HTTP only (production retrieval-service URL is what the
    prod proxy uses); the harness configures the request, the
    production service runs the pipeline. This keeps the search axis
    "what to ask the retrieval service" rather than "what code runs
    inside it" — cleaner, more reproducible, no secondary index drift.

12. **subject_entity attribution recovery**: closing the 98.3% unknown
    gap is the highest-leverage retrieval improvement. *Open: should
    the harness do this at retrieval time (slower per call) or at
    candidate-instantiation time (slower startup, faster per call)?*
    Default proposed: per-candidate startup pass that builds an
    `unknown_doc_id → recovered_subject_entity` map from the canonical
    JSONL once, applied to retrieval results in-flight. Map is per-run,
    not committed to the canonical store.

13. **Retrieval-bench cadence**: run 50-query bench every iteration
    (~$0.40 each), or once at end? *Default: every iteration. The
    proposer needs per-intent metrics to make per-intent policy
    decisions, and $0.40 × 5 = $2 is well within budget.*

---

## Next steps (after sign-off)

1. Lock the spec and tag it `meta-harness-domain-spec-v2`.
2. Build `lib/harness.py` Protocol + `lib/citations.py` + `lib/corpus_vocab.py`.
3. Build `agents/baseline_spark.py`. Reproduce the 94.4 baseline (acceptance test).
4. Build `lib/trace_recorder.py` + `lib/benchmark_response.py`.
5. Build `lib/benchmark_retrieval.py` (lift from audit's run5 driver).
6. Author the proposer skill at `.claude/skills/meta-harness-acti/SKILL.md`.
7. Run iteration 0 (baselines only) on the pod, eyeball traces.
8. Run iterations 1–5, report deltas + open issues.
