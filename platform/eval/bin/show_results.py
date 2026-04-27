#!/usr/bin/env python3
"""Render a recent run's summary + (optionally) a per-scenario detail table.

Usage:
  show_results.py [RUN_ID]            # latest if RUN_ID omitted
  show_results.py --runs-dir DIR
  show_results.py --detailed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_RUNS_DIR = Path(os.environ.get("ACTI_EVAL_RUNS_DIR", "/opt/acti/eval/runs"))


def _latest_run(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    return candidates[-1] if candidates else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_id", nargs="?", default=None)
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--detailed", action="store_true",
                   help="Print per-scenario score and any CRITICAL failures.")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    if args.run_id:
        run_dir = runs_dir / args.run_id
    else:
        run_dir = _latest_run(runs_dir)
        if run_dir is None:
            print(f"no runs found under {runs_dir}", file=sys.stderr)
            return 2

    summary_md = run_dir / "summary.md"
    if summary_md.exists():
        print(summary_md.read_text())
    else:
        print(f"no summary.md in {run_dir}", file=sys.stderr)

    if args.detailed:
        print("\n## Per-scenario detail\n")
        results = run_dir / "results.jsonl"
        if not results.exists():
            print("no results.jsonl", file=sys.stderr)
            return 2
        rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
        rows.sort(key=lambda r: r["score_0_100"])
        print("| id | intent | register | score | crit fails | latency | error |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            crit = ",".join(r.get("failed_critical") or []) or "-"
            err = (r.get("sut_error") or "-")[:40]
            print(f"| {r['scenario_id']} | {r['intent']} | "
                  f"{r.get('register', '?')} | {r['score_0_100']} | "
                  f"{crit} | {r.get('latency_s', '?')}s | {err} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
