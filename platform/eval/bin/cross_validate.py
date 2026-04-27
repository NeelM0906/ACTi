#!/usr/bin/env python3
"""Cross-validate Sohn-as-judge against an external judge (OpenRouter).

Picks N scenarios from a previous run, re-judges each one with both Sohn and
the external model, and reports per-dimension agreement.

Usage:
  cross_validate.py [--run-id RUN_ID] [--n 10] [--seed 42]
                    [--external-base-url https://openrouter.ai/api/v1]
                    [--external-model anthropic/claude-opus-4.7]
                    [--external-api-key-env OPENROUTER_API_KEY]
                    [--out-file PATH]

Strategy:
  - Load run_id's results.jsonl (if RUN_ID omitted, use latest).
  - Stratified pick across intents: 2 explain-teaching, 2 person-recall,
    1 case-lookup, 1 kai-memory, 1 continuity-snapshot, 1 general, 2 adversarial.
    Skip scenarios with sut_error or empty response_text.
  - For each, take the existing Sohn judgement from the run; ask the external
    model with the same judge prompt; parse and compare.
  - Output: cross_judge_<external_model>.jsonl per-scenario, plus a
    summary table written to stdout AND the run dir.

Agreement metric: per-dim, the fraction of scenarios where both judges gave
the same score. Discrepancies are reported with both scores and reasons so
a human can adjudicate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from lib import judge as _judge          # noqa: E402
from lib import runner as _runner        # noqa: E402


DEFAULT_RUNS_DIR = Path(os.environ.get("ACTI_EVAL_RUNS_DIR", "/opt/acti/eval/runs"))


# Stratified pick — exact counts per intent. Tweak only if you know what you're doing.
STRATIFIED_PICK = {
    "explain-teaching":    2,
    "person-recall":       2,
    "case-lookup":         1,
    "kai-memory":          1,
    "continuity-snapshot": 1,
    "general":             1,
    # adversarial scenarios all carry intent=general per the YAML (with
    # ad-* ids). Pick 2 of those by id-prefix.
    "_adversarial":        2,
}


def _pick_scenarios(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_intent: dict[str, list[dict]] = defaultdict(list)
    by_id_prefix: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("sut_error") or not r.get("response_text", "").strip():
            continue
        sid = r["scenario_id"]
        prefix = sid.split("-", 1)[0]
        by_id_prefix[prefix].append(r)
        # Adversarials are id-prefix ad-*; bucket them separately.
        if prefix == "ad":
            continue
        by_intent[r.get("intent", "?")].append(r)

    picked: list[dict] = []
    for intent, n in STRATIFIED_PICK.items():
        if intent == "_adversarial":
            pool = by_id_prefix.get("ad", [])
        else:
            pool = by_intent.get(intent, [])
        if not pool:
            continue
        rng.shuffle(pool)
        picked.extend(pool[:n])
    return picked


def _latest_run(runs_dir: Path) -> Path | None:
    candidates = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    return candidates[-1] if candidates else None


def _load_results(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with (run_dir / "results.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _agreement(sohn_scores: dict, ext_scores: dict) -> dict[str, dict]:
    """Per-dimension agreement summary. Both judges scored the same dim:
    counted as agree if scores match exactly, partial if differ by 1, disagree if differ by 2.
    """
    out: dict[str, dict] = defaultdict(lambda: {"agree": 0, "partial": 0, "disagree": 0,
                                                "sohn_higher": 0, "ext_higher": 0,
                                                "samples": []})
    for did, s in sohn_scores.items():
        e = ext_scores.get(did)
        if not e:
            continue
        ss = s.get("score")
        es = e.get("score")
        if ss in (None, -1) or es in (None, -1):
            continue
        if ss == es:
            out[did]["agree"] += 1
        elif abs(ss - es) == 1:
            out[did]["partial"] += 1
            (out[did]["sohn_higher" if ss > es else "ext_higher"]) += 1
        else:
            out[did]["disagree"] += 1
            (out[did]["sohn_higher" if ss > es else "ext_higher"]) += 1
        out[did]["samples"].append((ss, es))
    return out


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None)
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--rubric", default=str(HERE / "rubric.yaml"))
    p.add_argument("--n", type=int, default=10,
                   help="Total samples (must match sum of stratified pick).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--external-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--external-model", default="anthropic/claude-opus-4.7")
    p.add_argument("--external-api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--out-file", default=None)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    run_dir = (runs_dir / args.run_id) if args.run_id else _latest_run(runs_dir)
    if run_dir is None or not run_dir.exists():
        print(f"ERROR: run dir not found", file=sys.stderr)
        return 2
    print(f"using run: {run_dir.name}")

    rubric = _judge.load_rubric(Path(args.rubric))
    ext_key = os.environ.get(args.external_api_key_env)
    if not ext_key:
        print(f"ERROR: ${args.external_api_key_env} not set", file=sys.stderr)
        return 2

    rows = _load_results(run_dir)
    sample = _pick_scenarios(rows, args.seed)
    print(f"sampled {len(sample)} scenarios: " +
          ", ".join(r["scenario_id"] for r in sample))
    if not sample:
        return 2

    # Re-judge each with the external model. Synthesize a scenario dict
    # carrying the runtime flags that the runner enriched.
    out_rows: list[dict] = []
    for r in sample:
        scen = {
            "id": r["scenario_id"],
            "intent": r.get("intent"),
            "register": r.get("register"),
            "user_prompt": "(see scenario file)",
            "expects_corpus_grounding": bool(r.get("n_hits", 0)),
            "_recall_called": any(
                (tc.get("function") or {}).get("name") == "recall_context"
                for tc in r.get("tool_calls", [])
            ),
            "_tool_calls": r.get("tool_calls", []),
        }
        # Reload the original scenario fields (register, emotional, etc.) from the
        # YAML so applicable_dimensions matches the original run.
        scenarios = _runner.load_scenarios(HERE / "scenarios")
        by_id = {s["id"]: s for s in scenarios}
        if r["scenario_id"] in by_id:
            orig = by_id[r["scenario_id"]]
            for k in ("register", "emotional", "expects_corpus_grounding",
                     "expected_canonical_identity_line", "expected_tool_args",
                     "user_prompt"):
                if k in orig:
                    scen[k] = orig[k]

        # Build sample retrieved_hits for the judge (first 5 of each tool call).
        hits: list[dict] = []
        for tc in r.get("tool_calls", []) or []:
            hits.extend(tc.get("hits", [])[:5])

        print(f"  judging {scen['id']} via {args.external_model} ...", flush=True)
        ext_result = await _judge.judge_response(
            rubric=rubric, scenario=scen, response_text=r["response_text"],
            retrieved_hits=hits, tool_calls=r.get("tool_calls", []),
            base_url=args.external_base_url, api_key=ext_key,
            model=args.external_model, extra_body=None,
            cache_namespace="opus",
            use_cache=True,
        )

        sohn_scores = (r.get("judgement") or {}).get("scores", {})
        ext_scores = ext_result.to_dict()["scores"]
        out_rows.append({
            "scenario_id": r["scenario_id"],
            "intent": r.get("intent"),
            "register": r.get("register"),
            "sohn_judge": sohn_scores,
            "external_judge": ext_scores,
            "external_error": ext_result.error,
        })

    # Aggregate
    agg = defaultdict(lambda: {"agree": 0, "partial": 0, "disagree": 0,
                               "sohn_higher": 0, "ext_higher": 0, "n": 0})
    discrepancies: list[dict] = []
    for row in out_rows:
        a = _agreement(row["sohn_judge"], row["external_judge"])
        for did, stats in a.items():
            agg[did]["agree"]    += stats["agree"]
            agg[did]["partial"]  += stats["partial"]
            agg[did]["disagree"] += stats["disagree"]
            agg[did]["sohn_higher"] += stats["sohn_higher"]
            agg[did]["ext_higher"]  += stats["ext_higher"]
            agg[did]["n"] += stats["agree"] + stats["partial"] + stats["disagree"]
            for ss, es in stats["samples"]:
                if ss != es:
                    discrepancies.append({
                        "scenario_id": row["scenario_id"],
                        "dim": did,
                        "sohn": ss,
                        "ext": es,
                        "sohn_reason": (row["sohn_judge"].get(did, {}).get("reasoning") or "")[:200],
                        "ext_reason": (row["external_judge"].get(did, {}).get("reasoning") or "")[:200],
                    })

    out_path = Path(args.out_file) if args.out_file else (run_dir / f"cross_judge_{args.external_model.replace('/', '_')}.jsonl")
    with out_path.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}\n")

    print("# Per-dimension agreement\n")
    print(f"| Dimension | n | Exact | Partial (±1) | Disagree (±2) | Sohn higher | Ext higher |")
    print(f"|---|---|---|---|---|---|---|")
    for did in sorted(agg):
        s = agg[did]
        if s["n"] == 0:
            continue
        pct_exact = round(100 * s["agree"] / s["n"], 0)
        print(f"| {did} | {s['n']} | {s['agree']} ({pct_exact}%) | {s['partial']} | {s['disagree']} | {s['sohn_higher']} | {s['ext_higher']} |")

    if discrepancies:
        print("\n# Discrepancies (judges disagreed by ≥1)\n")
        for d in discrepancies[:30]:
            print(f"- **{d['scenario_id']} / {d['dim']}**: Sohn={d['sohn']}, Ext={d['ext']}")
            print(f"  - Sohn: {d['sohn_reason']}")
            print(f"  - Ext:  {d['ext_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
