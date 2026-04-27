#!/usr/bin/env python3
"""Run the Sohn benchmark.

Usage:
  run_benchmark.py [--scenarios DIR] [--rubric PATH] [--system-prompt PATH]
                   [--sut-base-url URL] [--retrieval-base-url URL]
                   [--judge-base-url URL]
                   [--runs-dir DIR] [--judge-cache-dir DIR]
                   [--concurrency N] [--no-judge] [--no-cache]
                   [--coverage]

Env vars (override-able by flags):
  ACTI_EVAL_API_KEY        — bearer token for SUT + judge
  ACTI_LIBRARY_API_KEY     — bearer token for the retrieval service
  ACTI_EVAL_SUT_BASE_URL   — default http://127.0.0.1:8888/raw/v1
  ACTI_LIBRARY_BASE_URL    — default https://acti-retrieval-production.up.railway.app
  ACTI_EVAL_JUDGE_BASE_URL — defaults to SUT base url
  ACTI_EVAL_RUNS_DIR       — default /opt/acti/eval/runs
  ACTI_EVAL_CACHE_DIR      — default /opt/acti/eval/cache/judge
  ACTI_SYSTEM_PROMPT_PATH  — default /opt/acti/system_prompts/sohn.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running from the repo without pip-installing.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from lib import judge as _judge          # noqa: E402
from lib import runner as _runner        # noqa: E402
from lib import coverage as _coverage    # noqa: E402


def _abs(p: Path) -> Path:
    return p if p.is_absolute() else (Path.cwd() / p)


def main() -> int:
    p = argparse.ArgumentParser(description="Run the Sohn behavioral benchmark.")
    p.add_argument("--scenarios", default=str(HERE / "scenarios"))
    p.add_argument("--rubric", default=str(HERE / "rubric.yaml"))
    p.add_argument("--system-prompt",
                   default=os.environ.get("ACTI_SYSTEM_PROMPT_PATH",
                                          str(HERE.parent / "system_prompts" / "sohn.txt")))
    p.add_argument("--sut-base-url", default=_runner.DEFAULT_SUT_BASE_URL)
    p.add_argument("--retrieval-base-url", default=_runner.DEFAULT_RETRIEVAL_BASE_URL)
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--runs-dir", default=str(_runner.DEFAULT_RUNS_DIR))
    p.add_argument("--judge-cache-dir", default=str(_judge.DEFAULT_CACHE_DIR))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--no-judge", action="store_true",
                   help="Skip the LLM-as-judge pass; programmatic checks only.")
    p.add_argument("--no-cache", action="store_true",
                   help="Bypass the judge cache (always re-judge).")
    p.add_argument("--coverage", action="store_true",
                   help="Compute corpus coverage from retrieved hits and "
                        "write coverage.md to the run dir.")
    p.add_argument("--filter", default=None,
                   help="Glob-style filter on scenario id (e.g., 'pr-*').")
    args = p.parse_args()

    scenarios_dir = _abs(Path(args.scenarios))
    rubric_path = _abs(Path(args.rubric))
    system_prompt_path = _abs(Path(args.system_prompt))
    runs_dir = _abs(Path(args.runs_dir))
    judge_cache_dir = _abs(Path(args.judge_cache_dir))

    if not rubric_path.exists():
        print(f"ERROR: rubric not found: {rubric_path}", file=sys.stderr)
        return 2
    if not scenarios_dir.is_dir():
        print(f"ERROR: scenarios dir not found: {scenarios_dir}", file=sys.stderr)
        return 2
    if not system_prompt_path.exists():
        print(f"ERROR: system prompt not found: {system_prompt_path}", file=sys.stderr)
        return 2

    rubric = _judge.load_rubric(rubric_path)
    scenarios = _runner.load_scenarios(scenarios_dir)
    if args.filter:
        from fnmatch import fnmatch
        scenarios = [s for s in scenarios if fnmatch(s["id"], args.filter)]
    if not scenarios:
        print("ERROR: no scenarios matched", file=sys.stderr)
        return 2

    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    print(f"loaded rubric v{rubric['version']} | {len(rubric['dimensions'])} dims | "
          f"{len(scenarios)} scenarios | system prompt {len(system_prompt)} chars")

    report = asyncio.run(_runner.run_benchmark(
        scenarios=scenarios, rubric=rubric, system_prompt=system_prompt,
        sut_base_url=args.sut_base_url,
        retrieval_base_url=args.retrieval_base_url,
        judge_base_url=args.judge_base_url,
        runs_dir=runs_dir, judge_cache_dir=judge_cache_dir,
        concurrency=args.concurrency,
        use_judge=not args.no_judge, use_judge_cache=not args.no_cache,
    ))

    if args.coverage:
        run_dir = runs_dir / report.run_id
        rows = []
        with (run_dir / "results.jsonl").open() as f:
            for line in f:
                rows.append(json.loads(line))
        cov = _coverage.compute_coverage(rows)
        (run_dir / "coverage.md").write_text(_coverage.render_coverage_md(cov))
        (run_dir / "coverage.json").write_text(json.dumps({
            "n_scenarios": cov.n_scenarios,
            "distinct_hit_ids": cov.distinct_hit_ids,
            "namespace_distribution": cov.namespace_distribution,
            "blind_spots": cov.blind_spots,
            "near_duplicate_pairs": [
                {"a": a, "b": b, "jaccard": j}
                for (a, b, j) in cov.near_duplicate_pairs
            ],
            "redundant_scenarios": cov.redundant_scenarios,
            "diversity_mean_jaccard": cov.diversity_mean_jaccard,
        }, indent=2))
        report.coverage = cov
        print(f"wrote {run_dir}/coverage.md")

    print(f"\n=== RUN {report.run_id} ===")
    print(json.dumps(report.summary, indent=2, ensure_ascii=False))
    return 0 if report.summary.get("overall_status") != "RED" else 1


if __name__ == "__main__":
    sys.exit(main())
