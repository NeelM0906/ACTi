---
name: meta-harness-acti
description: Use when running iteration N of the ACTi-Sohn Meta-Harness loop. Reads past traces + frontier, proposes 1-2 new candidate harnesses targeting the weakest dimensions, writes them under `agents/`, and emits `pending_eval.json` for the runner to benchmark.
---

# Meta-Harness ACTi-Sohn — Proposer rules of the game

You are the **proposer** in an automated harness-search loop for the
ACTi/Unblinded Sohn agent. You are NOT improving the base model; you
are evolving the *code around* the base model — the agent loop, the
retrieval policy, the cortex policy, the citation discipline, the
register-switch heuristics.

## Required reading (read these first, every iteration)

1. `domain_spec.md` — full spec: search axes, eval surface, baseline.
2. `agents/baseline_spark.py` — the parent. Every candidate descends
   from this code (literally — copy + modify, don't write from scratch).
3. `runs/<this_run>/frontier_val.json` — current best by score.
4. `runs/<this_run>/evolution_summary.jsonl` — every past candidate's
   score + hypothesis + diff vs parent.
5. `runs/<this_run>/traces/iter_<N-1>/<best_candidate>/response/<scenario>/trial_00/`
   — pick 3-5 scenarios where the frontier candidate scored <100, read
   their `transcript.txt` + `score.json`. **Do not skim — read the
   actual responses + tool calls.**
6. `../../eval/rubric.yaml` — what "good" looks like, dimension-by-dimension.
7. `../../eval/scenarios/*.yaml` — what's in the search set vs held out.

## Hard constraints

- **Search the harness, not the base model.** Don't change the
  embedding model, the inference engine, the system prompt's *persona*
  voice (rules-of-the-game, not the voice). You CAN edit the system
  prompt, but the existing voice/Formula/identity rules must stay.
- **No new tools.** The toolset is fixed at `recall_context`. You can
  rewrite how it's dispatched, validated, or post-processed, but you
  cannot add `web_search`, `python`, or anything else.
- **Don't break baseline_spark.py.** It's the canonical parent. Copy it
  to `agents/<your_candidate>.py`, modify, leave the original alone.
- **Test interface compliance** before writing pending_eval.json: each
  candidate must `from agents.<name> import AgentHarness` and the class
  must satisfy the `lib.harness.ActiHarness` Protocol.
- **Score on programmatic checks first.** If your candidate fails the
  identity-lockdown smoke test (vendor name leak on `ad-001`), the
  runner will reject it before benchmarking. Run a manual check before
  submitting.

## Soft preferences

- **Bias toward structural changes**, not pure prompt edits. The diff
  log captures whether your change was prompt-only — flagged proposals
  get scrutinized harder.
- **One change per candidate.** If you have two ideas, ship two
  candidates, not one combined. Attribution matters.
- **Hypothesize, then test.** Each `pending_eval.json` candidate has a
  `hypothesis` field. Write the hypothesis BEFORE writing the code, and
  make the code the smallest change that tests the hypothesis.

## High-leverage targets (from the corpus audit + run5 retrieval bench)

These are concrete proposals informed by the canonical-store audit
under `~/Desktop/ACTi_base/pinecone-audit/` (see `domain_spec.md`
"Corpus reality" section for full data):

1. **Per-intent retrieval policy.** The audit's run5 benchmark shows
   rerank gives **+40 pp Hit@1 on `teaching-explain`** but
   **+0 pp on `case-lookup`** while exploding p95 from 383 ms → 7.7 s.
   Dispatch retrieval config per intent.

2. **Subject-entity attribution recovery.** 24,624 of 25,052 records
   in the `users` namespace have `subject_entity = "unknown"`. A
   post-retrieval pass that recovers slugs from `source.title` /
   `source.uri` / chunk text would unlock proper person-recall
   filtering.

3. **chunk_id-grounded citation enforcement.** Every chunk_id is
   deterministic and parseable (`{content_type}:{slug}:{hash12}:{idx4}`).
   Use `lib/citations.py` to programmatically validate every cited
   bracketed tag in the response. If a citation doesn't resolve to a
   returned hit, strip it + warn the user — don't let fabrications
   land.

4. **Topic-filtered teaching retrieval.** For substantive teaching
   queries, classify the topic from query text (40-tag vocab in
   `lib/corpus_vocab.py`), apply `filter_overrides={"topics": {...}}`
   on the recall_context call.

5. **Stuck Kai-Mastery-Training fabrication suppression.** km-001 /
   km-004 are documented as prompt-resistant. A programmatic guard:
   if a returned hit's `source_title` is a known placeholder (`?`,
   `kai`, etc.), drop it before formatting for the model.

## What to write

When you're ready to propose, write `runs/<this_run>/pending_eval.json`:

```json
{
  "candidates": [
    {
      "name": "iter1_per_intent_rerank",
      "import_path": "agents.iter1_per_intent_rerank:AgentHarness",
      "hypothesis": "Rerank=False on case-lookup cuts p95 latency and equals Hit@1 (run5 §case-lookup); rerank=True elsewhere preserves +40pp on teaching-explain.",
      "changes": "1-line override of recall_context handler: when intent='case-lookup', set top_k=5 and don't issue secondary rerank request.",
      "parent": "baseline_spark"
    }
  ]
}
```

Then STOP. The runner picks up `pending_eval.json` automatically.

## Anti-patterns (will get rejected)

- Multi-paragraph prose in the response. **Don't change Sohn's voice.**
- Adding tools beyond `recall_context`.
- Vendoring `acti_retrieval` source. Use HTTP only (per Q11 in the spec).
- Touching `lib/` or `meta_harness.py`. Those are infra, not search axes.
- Submitting without running the candidate against `ad-001`,
  `gn-001`, and `pr-001` locally first.

## When to give up

If 3 consecutive iterations show no improvement and your hypotheses
are all variations on prompt edits, surface that as the iteration's
report (`runs/<run>/reports/iter_N.md`): "Proposer is stuck —
recommended action: pause search, address corpus-side issues
(subject_entity attribution, Kai placeholder titles, topic-tag
density) which are upstream of harness search."

The whole point of Meta-Harness is to find harness improvements.
When there aren't any left, say so.
