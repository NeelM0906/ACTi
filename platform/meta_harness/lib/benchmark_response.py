"""Drive an ActiHarness candidate through the response-quality eval.

Reuses `platform/eval/lib/{judge,checks}` for scoring and the existing
14-dim rubric. Adds:
  - per-trial trace persistence via trace_recorder
  - per-candidate aggregate suitable for the meta-harness frontier

Public API:
    run_response_benchmark(harness, scenarios, rubric, *, runs_root, ...)
        -> CandidateReport
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Protocol

# ---------- import eval lib ----------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_LIB = _REPO_ROOT / "eval" / "lib"
for _p in (_EVAL_LIB, Path("/opt/acti/eval/lib"), Path(os.environ.get("ACTI_EVAL_LIB_DIR", "/nonexistent"))):
    if _p.is_dir() and (_p / "judge.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
else:
    raise ImportError(
        f"could not find eval/lib in any of {[str(_EVAL_LIB), '/opt/acti/eval/lib']}"
    )

import checks as _checks  # type: ignore[import-not-found]   # noqa: E402
import judge as _judge    # type: ignore[import-not-found]   # noqa: E402

# ---------- harness lib (siblings) ----------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.trace_recorder import (  # type: ignore[import-not-found]   # noqa: E402
    ScoreSummary,
    TraceLocator,
    record_trace,
)


# ---------- public types ----------


class HarnessLike(Protocol):
    """Subset of ActiHarness that this benchmark exercises."""

    name: str

    def run(
        self, user_query: str, user_id: str
    ) -> Awaitable[tuple[str, dict[str, Any]]]: ...


@dataclass
class ScenarioReport:
    scenario_id: str
    response: str                           # last trial's response (for display)
    score_0_100: float                      # mean across trials
    per_dim: dict[str, int]                 # last trial's per-dim (judge variance is 0 → stable)
    failed_critical: list[str]              # union across trials (any trial RED → reported)
    judge_rationales: dict[str, str]
    programmatic: dict[str, str]
    trace_dir: Path | None
    duration_s: float                       # sum across trials
    n_trials: int = 1
    trial_scores: list[float] = field(default_factory=list)
    trial_score_std: float = 0.0


@dataclass
class CandidateReport:
    candidate: str
    iteration: int
    n_scenarios: int
    overall_score_0_100: float
    overall_status: str                     # RED / YELLOW / GREEN
    dim_means: dict[str, float]
    intent_means: dict[str, float]
    critical_failures_by_dim: dict[str, int]
    high_dim_warnings: list[str]
    scenarios: list[ScenarioReport] = field(default_factory=list)


# ---------- rubric / scoring helpers ----------


def load_rubric(path: Path | str | None = None) -> dict:
    p = Path(path) if path else (_REPO_ROOT / "eval" / "rubric.yaml")
    return _judge.load_rubric(p)


def load_scenarios(scenarios_dir: Path | str | None = None) -> list[dict]:
    """Load all scenarios from yaml files in the dir (mirrors runner.load_scenarios)."""
    import yaml  # local import — eval lib already requires it

    d = Path(scenarios_dir) if scenarios_dir else (_REPO_ROOT / "eval" / "scenarios")
    out: list[dict] = []
    for path in sorted(d.glob("*.yaml")):
        if path.name.startswith("._"):
            continue
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "scenarios" not in data:
            continue
        for sc in data["scenarios"]:
            if isinstance(sc, dict) and sc.get("id"):
                out.append(sc)
    return out


def _critical_dims(rubric: dict) -> set[str]:
    return {
        d["id"] for d in rubric.get("dimensions", [])
        if d.get("priority") == "CRITICAL"
    }


def _weight(rubric: dict, dim_id: str) -> int:
    weights = rubric.get("weights", {})
    by_id = {d["id"]: d for d in rubric.get("dimensions", [])}
    pri = by_id.get(dim_id, {}).get("priority", "MED")
    return int(weights.get(pri, 1))


def _score_one(
    rubric: dict, scores: dict[str, int]
) -> tuple[float, list[str]]:
    """Weighted normalized 0..100 + list of CRITICAL dims that scored 0."""
    crits = _critical_dims(rubric)
    total_weight = 0
    total_value = 0
    failed: list[str] = []
    for did, sc in scores.items():
        if sc == -1:
            continue
        w = _weight(rubric, did)
        total_weight += w * 2
        total_value += w * sc
        if did in crits and sc == 0:
            failed.append(did)
    if total_weight == 0:
        return 0.0, failed
    return round(100.0 * total_value / total_weight, 1), failed


def _summarize(rubric: dict, reports: list[ScenarioReport]) -> dict:
    by_dim: dict[str, list[int]] = {}
    by_intent: dict[str, list[float]] = {}
    crit_failures: dict[str, int] = {}
    for r in reports:
        intent = r.scenario_id.split("-", 1)[0] if r.scenario_id else "?"
        by_intent.setdefault(intent, []).append(r.score_0_100)
        for did, sc in r.per_dim.items():
            if sc == -1:
                continue
            by_dim.setdefault(did, []).append(sc)
        for did in r.failed_critical:
            crit_failures[did] = crit_failures.get(did, 0) + 1
    dim_means = {did: round(statistics.mean(vs), 2) for did, vs in by_dim.items()}
    intent_means = {iv: round(statistics.mean(vs), 1) for iv, vs in by_intent.items()}
    overall = (
        round(statistics.mean([r.score_0_100 for r in reports]), 1)
        if reports else 0.0
    )

    gates = rubric.get("gates", {})
    warn_threshold = float(gates.get("warn_if_high_dim_avg_below", 1.5))
    high_dims = {
        d["id"] for d in rubric.get("dimensions", [])
        if d.get("priority") == "HIGH"
    }
    high_dim_warnings = [
        did for did, mean in dim_means.items()
        if did in high_dims and mean < warn_threshold
    ]
    overall_status = "GREEN"
    if crit_failures:
        overall_status = "RED"
    elif high_dim_warnings:
        overall_status = "YELLOW"

    return {
        "overall_score_0_100": overall,
        "overall_status": overall_status,
        "dim_means": dim_means,
        "intent_means": intent_means,
        "crit_failures": crit_failures,
        "high_dim_warnings": high_dim_warnings,
    }


# ---------- per-scenario driver ----------


async def _score_scenario(
    rubric: dict,
    scenario: dict,
    response: str,
    trace: dict[str, Any],
    *,
    judge_base_url: str | None,
    judge_api_key: str | None,
    judge_cache_dir: Path | None,
    judge_model: str | None,
    use_judge: bool,
) -> tuple[dict[str, int], dict[str, str], dict[str, str], list[str]]:
    """Run programmatic + (optionally) judge dims; return per-dim 0/1/2/-1 scores.

    Returns (per_dim, judge_rationales, programmatic_explanations, failed_critical).
    """
    # The intent_and_param_correctness check reads `scenario['_tool_calls']`
    # to score the recall_context call. The eval runner injects this; we
    # mirror that by translating our trace.tool_calls into the expected
    # OpenAI-shape so the upstream check works unchanged.
    scenario_with_calls = dict(scenario)
    scenario_with_calls["_tool_calls"] = [
        {
            "function": {"name": tc.get("name", "")},
            "args": tc.get("arguments_parsed") or {},
        }
        for tc in (trace.get("tool_calls") or [])
    ]
    # 1. Programmatic checks. CheckResult.score can be None (=abstain → judge);
    #    judge_response merges this for us.
    prog = _checks.run_programmatic_checks(rubric, scenario_with_calls, response)
    programmatic_explanations: dict[str, str] = {
        did: "; ".join(res.evidence) if res.evidence else (
            "abstain" if res.score is None else f"score={res.score}"
        )
        for did, res in prog.items()
    }

    per_dim: dict[str, int] = {}
    judge_rationales: dict[str, str] = {}

    if use_judge:
        # judge_response handles applicability + merge + cache internally.
        jr = await _judge.judge_response(
            rubric=rubric,
            scenario=scenario_with_calls,
            response_text=response,
            retrieved_hits=trace.get("retrieval_hits") or [],
            tool_calls=trace.get("tool_calls") or [],
            programmatic=prog,
            base_url=judge_base_url or os.environ.get(
                "ACTI_EVAL_JUDGE_BASE_URL",
                "http://127.0.0.1:8080/raw/v1",
            ),
            api_key=judge_api_key or os.environ.get(
                "ACTI_EVAL_JUDGE_API_KEY",
                os.environ.get("ACTI_EVAL_API_KEY", ""),
            ),
            cache_dir=judge_cache_dir or Path("/tmp/acti-meta-harness/judge-cache"),
            model=judge_model or "Sohn",
        )
        for did, ds in jr.scores.items():
            per_dim[did] = ds.score
            if ds.reasoning:
                judge_rationales[did] = ds.reasoning
    else:
        # No judge — accept only programmatic-decided dims; rest are N/A.
        for did, res in prog.items():
            if res.score is not None:
                per_dim[did] = res.score

    # Fill in any applicable dim that didn't get scored as N/A.
    for did in _judge.applicable_dimensions(scenario):
        per_dim.setdefault(did, -1)

    crits = _critical_dims(rubric)
    failed = [did for did, sc in per_dim.items() if sc == 0 and did in crits]
    return per_dim, judge_rationales, programmatic_explanations, failed


# ---------- public entry point ----------


async def run_response_benchmark(
    *,
    harness: HarnessLike,
    scenarios: list[dict],
    rubric: dict,
    runs_root: Path,
    iteration: int = 0,
    trials: int = 1,
    use_judge: bool = True,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_cache_dir: Path | None = None,
    judge_model: str | None = None,
    concurrency: int = 1,
    log: callable = print,
) -> CandidateReport:
    """Drive the candidate through scenarios; score each; persist traces.

    `runs_root` is /opt/acti/meta_harness/runs/<run>/traces/. Caller is
    responsible for naming the run.

    `trials` controls per-scenario re-runs. The SUT has temperature > 0,
    so each trial's response varies. Mean-aggregating N trials reduces
    noise by √N. Default 1 (single-shot, fast). Use 3+ for benchmark
    runs where you want statistical signal.
    """
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    trials = max(1, int(trials))

    async def _run_one_trial(
        sc: dict, trial: int
    ) -> tuple[float, dict, dict, list[str], dict, dict, str, str, Path | None, float]:
        """Run one (scenario, trial) — returns the score + side data."""
        scenario_id = sc["id"]
        t0 = time.time()
        user_id = f"eval:{scenario_id}:t{trial:02d}"
        try:
            response, trace = await harness.run(sc["user_prompt"], user_id)
        except Exception as e:  # noqa: BLE001
            response = f"ERROR: harness raised: {e}"
            trace = {
                "candidate": getattr(harness, "name", "?"),
                "scenario_id": scenario_id,
                "user_id": user_id,
                "user_query": sc["user_prompt"],
                "final_response": response,
                "transcript": [],
                "tool_calls": [],
                "retrieval_hits": [],
                "harness_meta": {"error": str(e)},
            }
        duration = time.time() - t0

        per_dim, jr_expl, prog_expl, failed = await _score_scenario(
            rubric, sc, response, trace,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
            judge_cache_dir=judge_cache_dir,
            judge_model=judge_model,
            use_judge=use_judge,
        )
        score, _ = _score_one(rubric, per_dim)

        locator = TraceLocator(
            runs_root=runs_root,
            iteration=iteration,
            candidate=getattr(harness, "name", "candidate"),
            scenario_id=scenario_id,
            trial=trial,
        )
        summary = ScoreSummary(
            weighted_total=score,
            per_dim=per_dim,
            judge_rationale=jr_expl,
            programmatic=prog_expl,
            gate="RED" if failed else "GREEN",
            crit_violations=failed,
        )
        trace_dir = record_trace(locator, trace, summary)
        return (
            score, per_dim, jr_expl, failed, prog_expl, dict(trace),
            response, scenario_id, trace_dir, duration,
        )

    async def _one(sc: dict) -> ScenarioReport:
        scenario_id = sc["id"]
        async with sem:
            trial_scores: list[float] = []
            last_per_dim: dict[str, int] = {}
            last_jr: dict[str, str] = {}
            last_prog: dict[str, str] = {}
            failed_union: set[str] = set()
            last_response = ""
            last_trace_dir: Path | None = None
            total_duration = 0.0

            for trial in range(trials):
                score, per_dim, jr, failed, prog, _trace, resp, _sid, td, dur = (
                    await _run_one_trial(sc, trial)
                )
                trial_scores.append(score)
                last_per_dim = per_dim
                last_jr = jr
                last_prog = prog
                failed_union |= set(failed)
                last_response = resp
                last_trace_dir = td
                total_duration += dur

            mean_score = sum(trial_scores) / len(trial_scores) if trial_scores else 0.0
            std = statistics.pstdev(trial_scores) if len(trial_scores) > 1 else 0.0
            tag = "ok  " if not failed_union else "RED "
            tail = (
                f"±{std:>3.1f}" if trials > 1 else "    "
            )
            log(
                f"  {scenario_id:<14} mean={mean_score:>5.1f} {tail}  "
                f"{tag}({total_duration:.1f}s, n={trials})"
            )
            return ScenarioReport(
                scenario_id=scenario_id,
                response=last_response,
                score_0_100=mean_score,
                per_dim=last_per_dim,
                failed_critical=sorted(failed_union),
                judge_rationales=last_jr,
                programmatic=last_prog,
                trace_dir=last_trace_dir,
                duration_s=total_duration,
                n_trials=trials,
                trial_scores=trial_scores,
                trial_score_std=std,
            )

    log(f"running {getattr(harness, 'name', '?')} over {len(scenarios)} "
        f"scenarios (concurrency={concurrency}, trials={trials})")
    reports = await asyncio.gather(*[_one(sc) for sc in scenarios])

    summary = _summarize(rubric, reports)
    return CandidateReport(
        candidate=getattr(harness, "name", "?"),
        iteration=iteration,
        n_scenarios=len(reports),
        overall_score_0_100=summary["overall_score_0_100"],
        overall_status=summary["overall_status"],
        dim_means=summary["dim_means"],
        intent_means=summary["intent_means"],
        critical_failures_by_dim=summary["crit_failures"],
        high_dim_warnings=summary["high_dim_warnings"],
        scenarios=reports,
    )
