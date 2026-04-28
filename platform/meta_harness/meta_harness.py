"""Meta-Harness outer loop for ACTi-Sohn — Phase 0 + evolution iterations.

Patterned on Stanford's reference_examples/text_classification/meta_harness.py
but adapted for the ACTi domain spec (see ./domain_spec.md).

Phase 0 — Baselines:
  Run baseline_spark (and any other agents/baseline_*.py modules present)
  through the response benchmark on the search-set scenarios. Seeds the
  frontier_val.json. Reproduces the existing 94.4/100 eval baseline as
  the acceptance test.

Phase 1..N — Evolution:
  For each iteration:
    1. Render the proposer prompt (run dir, frontier, past results).
    2. Call Claude Code as the proposer. It reads filesystem context
       (logs, traces, scenarios, the rubric, this domain_spec.md), writes
       new candidate `.py` files under agents/, then writes pending_eval.json.
    3. Validate: import-check + smoke against ge-001 + identity check
       against ad-001.
    4. Benchmark valid candidates over the search set.
    5. Update frontier + evolution_summary.jsonl.

Phase Final — Held-out + retrieval bench:
  Run the frontier candidate on the held-out scenarios (×5 trials) +
  the full retrieval benchmark. Single shot, end of run.

Usage:
    python meta_harness.py --iterations 0           # Phase 0 only (baselines)
    python meta_harness.py --iterations 5 --fresh   # full 5-iter run
    python meta_harness.py --filter ad-* --iterations 0
    python meta_harness.py --no-judge --iterations 0   # cheap, programmatic only
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

# Ensure local imports work regardless of cwd.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib.benchmark_response import (  # noqa: E402
    CandidateReport,
    load_rubric,
    load_scenarios,
    run_response_benchmark,
)


# ---------- ANSI ----------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if _USE_COLOR else t


_bold = lambda t: _c("1", t)  # noqa: E731
_dim = lambda t: _c("2", t)
_red = lambda t: _c("31", t)
_green = lambda t: _c("32", t)
_yellow = lambda t: _c("33", t)
_cyan = lambda t: _c("36", t)


def _ts() -> str:
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(s: float) -> str:
    m, ss = divmod(int(s), 60)
    return f"{m}m{ss:02d}s" if m else f"{int(s)}s"


def _score_str(score: float) -> str:
    s = f"{score:.1f}"
    if score >= 90:
        return _green(s)
    if score >= 75:
        return _yellow(s)
    return _red(s)


# ---------- search-set / held-out split ----------

# Per domain_spec.md §Evaluation, stratified by intent prefix.
# IDs come from the existing platform/eval/scenarios/*.yaml.
HELD_OUT: frozenset[str] = frozenset({
    # explain-teaching: 3 of 8
    "et-006", "et-007", "et-008",
    # person-recall: 2 of 6
    "pr-005", "pr-006",
    # case-lookup: 2 of 5
    "cl-004", "cl-005",
    # kai-memory: 1 of 4
    "km-004",
    # continuity-snapshot: 1 of 4
    "cs-004",
    # general: 2 of 6
    "gn-005", "gn-006",
    # adversarial: 4 of 12
    "ad-009", "ad-010", "ad-011", "ad-012",
})


def split_scenarios(
    scenarios: list[dict], filter_glob: str | None = None
) -> tuple[list[dict], list[dict]]:
    """Return (search_set, held_out_set) per the canonical split.

    Optional `filter_glob` (e.g. 'ad-*') filters BOTH halves to only matching
    ids — useful for quick tests on a single bucket.
    """
    search: list[dict] = []
    held: list[dict] = []
    for sc in scenarios:
        sid = sc["id"]
        if filter_glob and not fnmatch(sid, filter_glob):
            continue
        if sid in HELD_OUT:
            held.append(sc)
        else:
            search.append(sc)
    return search, held


# ---------- candidate discovery ----------


def discover_candidates(agents_dir: Path) -> list[tuple[str, Path]]:
    """Return [(name, path), ...] for every .py file in agents/ that exposes
    AgentHarness at module scope.

    Order: baselines first (alphabetical by filename), then non-baselines.
    """
    out: list[tuple[str, Path]] = []
    for p in sorted(agents_dir.glob("*.py")):
        if p.name in ("__init__.py",):
            continue
        out.append((p.stem, p))
    # Baselines come first.
    out.sort(key=lambda x: (not x[0].startswith("baseline"), x[0]))
    return out


def load_agent_class(path: Path):
    """Import a candidate module and return its AgentHarness class."""
    name = f"acti_meta_harness_candidate_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    cls = getattr(mod, "AgentHarness", None)
    if cls is None:
        raise ImportError(f"{path} does not expose AgentHarness")
    return cls


# ---------- one-candidate phase ----------


async def benchmark_candidate(
    *,
    name: str,
    cls: type,
    scenarios: list[dict],
    rubric: dict,
    runs_root: Path,
    iteration: int,
    args: argparse.Namespace,
) -> CandidateReport:
    """Instantiate the candidate, run the response benchmark, return report."""
    engine_url = os.environ.get(
        "ACTI_HARNESS_ENGINE_BASE_URL", "http://127.0.0.1:8080/raw/v1"
    )
    api_key = os.environ.get("ACTI_EVAL_API_KEY", "")
    retrieval_url = os.environ.get(
        "ACTI_LIBRARY_BASE_URL", "https://acti-retrieval-production.up.railway.app"
    )
    retrieval_key = os.environ.get("ACTI_LIBRARY_API_KEY", "")
    if not retrieval_key:
        raise SystemExit("ACTI_LIBRARY_API_KEY is required (set in env)")

    harness = cls(
        engine_base_url=engine_url,
        retrieval_base_url=retrieval_url,
        api_key=api_key,
        retrieval_api_key=retrieval_key,
    )
    return await run_response_benchmark(
        harness=harness,
        scenarios=scenarios,
        rubric=rubric,
        runs_root=runs_root,
        iteration=iteration,
        trials=args.trials,
        use_judge=not args.no_judge,
        judge_cache_dir=Path(
            os.environ.get(
                "ACTI_HARNESS_JUDGE_CACHE_DIR",
                "/tmp/acti-meta-harness/judge-cache",
            )
        ),
        concurrency=args.concurrency,
        log=lambda line: print(_dim(f"  {line}"), flush=True),
    )


# ---------- frontier + summary ----------


def update_frontier(
    frontier_path: Path, candidate: str, report: CandidateReport
) -> bool:
    """Update frontier_val.json. Returns True if this candidate is the new best."""
    data = json.loads(frontier_path.read_text()) if frontier_path.exists() else {}
    best = data.get("_best", {})
    is_new_best = (
        report.overall_status != "RED"
        and report.overall_score_0_100 > best.get("overall_score_0_100", -1)
    )
    if is_new_best:
        data["_best"] = {
            "candidate": candidate,
            "overall_score_0_100": report.overall_score_0_100,
            "overall_status": report.overall_status,
            "iteration": report.iteration,
            "n_scenarios": report.n_scenarios,
            "dim_means": report.dim_means,
        }
    data.setdefault("history", []).append({
        "iteration": report.iteration,
        "candidate": candidate,
        "score": report.overall_score_0_100,
        "status": report.overall_status,
    })
    frontier_path.write_text(json.dumps(data, indent=2))
    return is_new_best


def append_evolution_summary(
    summary_path: Path,
    iteration: int,
    candidate: str,
    report: CandidateReport,
    propose_time_s: float | None,
    bench_time_s: float,
    is_new_best: bool,
) -> None:
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "candidate": candidate,
        "score_0_100": report.overall_score_0_100,
        "status": report.overall_status,
        "n_scenarios": report.n_scenarios,
        "dim_means": report.dim_means,
        "intent_means": report.intent_means,
        "crit_failures": report.critical_failures_by_dim,
        "high_dim_warnings": report.high_dim_warnings,
        "propose_time_s": round(propose_time_s, 1) if propose_time_s else None,
        "bench_time_s": round(bench_time_s, 1),
        "is_new_best": is_new_best,
    }
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------- phase 0 ----------


async def phase_0_baselines(
    *,
    candidates: list[tuple[str, Path]],
    scenarios: list[dict],
    rubric: dict,
    args: argparse.Namespace,
    run_dir: Path,
) -> list[tuple[str, CandidateReport]]:
    n_run = len(candidates) if args.iterations > 0 else len(
        [c for c in candidates if c[0].startswith("baseline")]
    )
    label = (
        f"Phase 0 — baselines ({n_run} baselines, {len(scenarios)} scenarios)"
        if args.iterations == 0
        else f"Phase 0+ — all candidates ({n_run}, {len(scenarios)} scenarios)"
    )
    print(f"\n{_ts()} {_bold(label)}")

    reports: list[tuple[str, CandidateReport]] = []
    frontier_path = run_dir / "frontier_val.json"
    summary_path = run_dir / "evolution_summary.jsonl"
    runs_root = run_dir / "traces"

    for name, path in candidates:
        # When --iterations 0 we run baselines only; otherwise we run every
        # candidate discovered in agents/ so iter-N proposals get scored.
        if args.iterations == 0 and not name.startswith("baseline"):
            continue
        print(f"  {_ts()} {_bold(name)}…", flush=True)
        try:
            cls = load_agent_class(path)
        except Exception as e:  # noqa: BLE001
            print(f"    {_red('IMPORT FAIL')}: {e}")
            continue
        t0 = time.time()
        try:
            report = await benchmark_candidate(
                name=name,
                cls=cls,
                scenarios=scenarios,
                rubric=rubric,
                runs_root=runs_root,
                iteration=0,
                args=args,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    {_red('BENCH FAIL')}: {e}")
            continue
        elapsed = time.time() - t0
        is_best = update_frontier(frontier_path, name, report)
        append_evolution_summary(
            summary_path, 0, name, report,
            propose_time_s=None, bench_time_s=elapsed, is_new_best=is_best,
        )
        status_color = (
            _green if report.overall_status == "GREEN"
            else (_yellow if report.overall_status == "YELLOW" else _red)
        )
        print(f"    {_ts()} {name}: score={_score_str(report.overall_score_0_100)} "
              f"status={status_color(report.overall_status)} "
              f"({_elapsed(elapsed)}) "
              f"{_green('NEW BEST') if is_best else ''}")
        if report.critical_failures_by_dim:
            print(f"      crit_failures: {dict(report.critical_failures_by_dim)}")
        if report.high_dim_warnings:
            print(f"      high_dim_warnings: {report.high_dim_warnings}")
        reports.append((name, report))

    return reports


# ---------- main entry ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="ACTi-Sohn Meta-Harness")
    parser.add_argument("--iterations", type=int, default=0,
                        help="0 = baselines only (Phase 0); N>0 enables proposer loop "
                             "(not yet implemented).")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="Clear logs and proposed agents (keeps baselines).")
    parser.add_argument("--filter", type=str, default=None,
                        help="fnmatch glob to filter scenarios (e.g. 'ad-*').")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM judge — programmatic dims only. Cheap and "
                             "deterministic for sanity checks.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallel scenarios. Default 1 (serial). 4 is safe on the pod.")
    parser.add_argument("--trials", type=int, default=1,
                        help="Trials per scenario. Score is the mean. Use 3+ for "
                             "benchmark runs (single-trial has ~±2 noise floor).")
    parser.add_argument("--runs-root", type=str, default=None,
                        help="Override default runs root (default $ACTI_HARNESS_RUNS_DIR "
                             "or /opt/acti/meta_harness/runs/).")
    args = parser.parse_args()

    # Resolve run dir.
    runs_root = Path(
        args.runs_root
        or os.environ.get("ACTI_HARNESS_RUNS_DIR")
        or "/opt/acti/meta_harness/runs"
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(exist_ok=True)

    if args.fresh:
        for f in (
            run_dir / "frontier_val.json",
            run_dir / "evolution_summary.jsonl",
            run_dir / "pending_eval.json",
        ):
            if f.exists():
                f.unlink()

    # Resolve scenarios + rubric.
    rubric = load_rubric()
    scenarios_all = load_scenarios()
    search_set, held_out = split_scenarios(scenarios_all, filter_glob=args.filter)

    print(f"{_ts()} {_bold('Meta-Harness ACTi-Sohn')} "
          f"run={_cyan(run_name)} "
          f"rubric_v{rubric['version']} "
          f"search={len(search_set)}/{len(scenarios_all)} "
          f"held_out={len(held_out)}")
    print(f"  run_dir: {run_dir}")
    print(f"  judge: {'OFF (programmatic-only)' if args.no_judge else 'ON'}")
    print(f"  concurrency: {args.concurrency}")

    candidates = discover_candidates(HERE / "agents")
    if not candidates:
        raise SystemExit("no candidates found in agents/")
    print(f"  candidates: {[c[0] for c in candidates]}")

    if not search_set:
        raise SystemExit("search_set is empty — check --filter")

    # ---- Phase 0 ----
    reports = asyncio.run(phase_0_baselines(
        candidates=candidates,
        scenarios=search_set,
        rubric=rubric,
        args=args,
        run_dir=run_dir,
    ))

    if args.iterations > 0:
        print(f"\n{_yellow('NOTE: iterations > 0 not yet implemented in this build')}")
        print(f"      Phase 0 baselines complete; proposer loop is a follow-up.")

    # Final summary
    print(f"\n{_ts()} {_bold('Phase 0 complete')}")
    if reports:
        print(f"  candidates run: {len(reports)}")
        scores = [r.overall_score_0_100 for _, r in reports]
        print(f"  score range: {min(scores):.1f} … {max(scores):.1f} "
              f"(mean={statistics.mean(scores):.1f})")
    print(f"  artifacts: {run_dir}")


if __name__ == "__main__":
    main()
