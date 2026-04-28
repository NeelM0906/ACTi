#!/usr/bin/env python3
"""Pretty-print the evolution_summary.jsonl for a Meta-Harness run.

Usage:
    python bin/show_run.py <run-name>
    python bin/show_run.py /path/to/runs/<run-name>/

Shows: per-candidate aggregate (score, status, dim means, intent means),
deltas vs baseline_spark, and the frontier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make `lib` importable for the future (currently this script doesn't need it).
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


# ---------- ANSI ----------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if _USE_COLOR else t


_bold = lambda t: _c("1", t)            # noqa: E731
_dim = lambda t: _c("2", t)
_red = lambda t: _c("31", t)
_green = lambda t: _c("32", t)
_yellow = lambda t: _c("33", t)
_cyan = lambda t: _c("36", t)


def _score_color(score: float) -> str:
    if score >= 96:
        return _green
    if score >= 90:
        return _yellow
    return _red


def _resolve_run_dir(run_arg: str) -> Path:
    """Accept either a bare run name (relative to default runs root) or a
    full path. Bare names are resolved against $ACTI_HARNESS_RUNS_DIR
    or /opt/acti/meta_harness/runs/."""
    p = Path(run_arg)
    if p.exists():
        return p
    root = Path(
        os.environ.get("ACTI_HARNESS_RUNS_DIR") or "/opt/acti/meta_harness/runs"
    )
    return root / run_arg


def _load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "evolution_summary.jsonl"
    if not path.exists():
        sys.exit(f"no evolution_summary.jsonl at {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _load_frontier(run_dir: Path) -> dict:
    path = run_dir / "frontier_val.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _print_header(run_dir: Path, rows: list[dict]) -> None:
    n = len(rows)
    print(_bold(f"Run: {run_dir.name}") + _dim(f"  ({run_dir})"))
    if rows:
        ts = rows[0].get("ts", "?")
        n_scenarios = rows[0].get("n_scenarios", "?")
        print(f"  candidates: {n}, scenarios per candidate: {n_scenarios}, "
              f"first ts: {ts}")


def _format_delta(delta: float, fmt: str = "{:+.1f}") -> str:
    s = fmt.format(delta)
    if delta > 0.001:
        return _green(s)
    if delta < -0.001:
        return _red(s)
    return _dim(s)


def _print_per_candidate(rows: list[dict]) -> None:
    """Score + per-dim + per-intent table, one row per candidate."""
    if not rows:
        return

    hdr_dims = (
        "citation_grounding", "no_emoji", "anti_sycophancy",
        "intent_and_param_correctness", "conciseness",
    )
    hdr_intents = ("ad", "et", "pr", "cl", "km", "cs", "gn")

    # Header
    print()
    print(_bold("Per-candidate (score / status / dim means / intent means):"))
    short_dims = {
        "citation_grounding": "cite",
        "no_emoji": "emoji",
        "anti_sycophancy": "anti",
        "intent_and_param_correctness": "intent",
        "conciseness": "conc",
    }
    cols = (
        ["candidate".ljust(34), "score", "stat"]
        + [short_dims[d] for d in hdr_dims]
        + list(hdr_intents)
    )
    sep = " | "
    print(_dim(sep.join(c.rjust(5) if i > 0 else c for i, c in enumerate(cols))))
    print(_dim("-" * (sum(max(len(c), 5) + 3 for c in cols))))

    for r in rows:
        name = r["candidate"][:34]
        score = r["score_0_100"]
        status = r["status"]
        dim_means = r.get("dim_means", {})
        intent_means = r.get("intent_means", {})
        score_c = _score_color(score)
        status_c = (
            _green if status == "GREEN"
            else (_yellow if status == "YELLOW" else _red)
        )
        line_parts = [
            name.ljust(34),
            score_c(f"{score:>5.1f}"),
            status_c(f"{status:>5}"),
        ]
        for d in hdr_dims:
            v = dim_means.get(d)
            line_parts.append(f"{v:>5.2f}" if v is not None else "  -- ")
        for it in hdr_intents:
            v = intent_means.get(it)
            line_parts.append(f"{v:>5.1f}" if v is not None else "  -- ")
        print(sep.join(line_parts))


def _print_deltas_vs_baseline(rows: list[dict]) -> None:
    """Per-dim deltas vs baseline_spark, hiding zero-deltas."""
    base = next((r for r in rows if r["candidate"] == "baseline_spark"), None)
    if base is None:
        return
    print()
    print(_bold("Deltas vs baseline_spark:"))
    base_dim = base.get("dim_means", {})
    base_score = base["score_0_100"]
    for r in rows:
        if r["candidate"] == "baseline_spark":
            continue
        print(f"  {_cyan(r['candidate'])}:")
        for k in (
            "citation_grounding", "no_emoji", "anti_sycophancy",
            "intent_and_param_correctness", "conciseness",
            "truth_over_comfort", "zone_action",
        ):
            d = r.get("dim_means", {}).get(k, 0) - base_dim.get(k, 0)
            if abs(d) > 0.001:
                print(f"    {k:<32}: {_format_delta(d, '{:+.2f}')}")
        score_d = r["score_0_100"] - base_score
        print(f"    {'OVERALL_score':<32}: {_format_delta(score_d)}")


def _print_critical_violations(rows: list[dict]) -> None:
    print()
    print(_bold("CRITICAL violations (count of (scenario, trial) pairs that scored 0):"))
    for r in rows:
        cv = r.get("crit_failures") or {}
        if not cv:
            print(f"  {_green('—')} {r['candidate']}: clean")
        else:
            print(f"  {_red('!')} {r['candidate']}: {dict(cv)}")


def _print_frontier(frontier: dict) -> None:
    if not frontier:
        return
    best = frontier.get("_best") or {}
    print()
    print(_bold("Frontier:"))
    if best:
        print(
            f"  best = {_cyan(best.get('candidate', '?'))}, "
            f"score = {_score_color(best.get('overall_score_0_100', 0))(f'{best.get(chr(39)+chr(34)+chr(34)+chr(39), 0):.1f}')[:-2]}"
            if False else
            f"  best = {_cyan(best.get('candidate', '?'))}, "
            f"score = {best.get('overall_score_0_100', 0):.1f}, "
            f"status = {best.get('overall_status', '?')}"
        )
    history = frontier.get("history") or []
    if history:
        print(f"  history: {len(history)} candidate runs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="Run name or full path to runs/<name>/")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run)
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")

    rows = _load_rows(run_dir)
    frontier = _load_frontier(run_dir)

    _print_header(run_dir, rows)
    _print_per_candidate(rows)
    _print_deltas_vs_baseline(rows)
    _print_critical_violations(rows)
    _print_frontier(frontier)


if __name__ == "__main__":
    main()
