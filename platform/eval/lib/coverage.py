"""Scenario coverage analysis using the sohn-embedding-sm index.

We don't get raw embedding vectors from the retrieval service (it only exposes
/v1/retrieve), so we use the retrieval response itself as a coverage proxy:

  - Each scenario triggers a retrieval call with intent + query.
  - The set of returned hit IDs is a "footprint" of which corpus regions the
    scenario touches.
  - Coverage = |union(footprints_i)| across the corpus, by namespace.
  - Diversity = pairwise Jaccard between footprints. <0.3 is diverse;
    >0.7 is near-duplicate.
  - Blind-spot detection = which namespaces have low or zero footprint coverage
    relative to the corpus shape (~50k chunks: teachings 36k, user 8.7k,
    memory 1k, cases 61, products 171, identity 7).

Public API:
    compute_coverage(reports) -> CoverageReport
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass


# Approximate corpus shape per library.py (used to surface blind spots when a
# namespace has near-zero scenario coverage relative to its size).
APPROX_NAMESPACE_SIZE = {
    "teachings": 36_000,
    "user": 8_700,
    "memory": 1_000,
    "cases": 61,
    "products": 171,
    "identity": 7,
}


@dataclass
class CoverageReport:
    n_scenarios: int
    distinct_hit_ids: int
    namespace_distribution: dict[str, int]   # hit count per namespace
    blind_spots: list[str]                   # namespace names with 0 coverage
    near_duplicate_pairs: list[tuple[str, str, float]]  # (id_a, id_b, jaccard)
    redundant_scenarios: list[str]           # scenarios involved in any pair >0.7
    diversity_mean_jaccard: float            # lower = more diverse


def _scenario_footprint(scenario_run_dict: dict) -> set[str]:
    """Footprint = set of hit IDs returned across all tool calls in the scenario.

    Accepts a dict with shape {"tool_calls": [{"hits": [{...}]}]} OR a runner
    ScenarioReport-shaped dict.
    """
    ids: set[str] = set()
    for tc in scenario_run_dict.get("tool_calls", []) or []:
        for h in tc.get("hits", []) or []:
            hid = h.get("id")
            if hid:
                ids.add(str(hid))
    return ids


def _namespace_distribution(scenario_runs: list[dict]) -> Counter:
    c: Counter = Counter()
    for sr in scenario_runs:
        for tc in sr.get("tool_calls", []) or []:
            for h in tc.get("hits", []) or []:
                ns = (h.get("metadata") or {}).get("namespace", "unknown")
                c[ns] += 1
    return c


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def compute_coverage(scenario_runs: list[dict]) -> CoverageReport:
    """Compute coverage + redundancy from a list of scenario-run dicts.

    Each input dict must have keys `scenario_id` and `tool_calls` (each call
    carrying `hits`). The runner.results.jsonl rows match this shape.
    """
    footprints: dict[str, set[str]] = {
        sr["scenario_id"]: _scenario_footprint(sr) for sr in scenario_runs
    }

    distinct_ids: set[str] = set()
    for fp in footprints.values():
        distinct_ids |= fp

    ns_dist = _namespace_distribution(scenario_runs)
    blind_spots = [
        ns for ns in APPROX_NAMESPACE_SIZE
        if ns_dist.get(ns, 0) == 0 and APPROX_NAMESPACE_SIZE[ns] >= 50
    ]

    pairs: list[tuple[str, str, float]] = []
    redundant_set: set[str] = set()
    sids = sorted(footprints.keys())
    jaccs: list[float] = []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            a, b = sids[i], sids[j]
            j_ = _jaccard(footprints[a], footprints[b])
            if footprints[a] or footprints[b]:
                jaccs.append(j_)
            if j_ > 0.7:
                pairs.append((a, b, round(j_, 2)))
                redundant_set.add(a)
                redundant_set.add(b)

    diversity = round(statistics.mean(jaccs), 3) if jaccs else 0.0

    return CoverageReport(
        n_scenarios=len(scenario_runs),
        distinct_hit_ids=len(distinct_ids),
        namespace_distribution=dict(ns_dist),
        blind_spots=blind_spots,
        near_duplicate_pairs=pairs,
        redundant_scenarios=sorted(redundant_set),
        diversity_mean_jaccard=diversity,
    )


def render_coverage_md(cov: CoverageReport) -> str:
    lines = ["# Scenario coverage", "",
             f"- Scenarios: {cov.n_scenarios}",
             f"- Distinct hits across all scenarios: {cov.distinct_hit_ids}",
             f"- Diversity (mean pairwise Jaccard): {cov.diversity_mean_jaccard} (lower = more diverse)",
             ""]
    lines.append("## Namespace distribution")
    lines.append("| Namespace | Hit count | Approx corpus size |")
    lines.append("|---|---|---|")
    for ns in sorted(set(list(cov.namespace_distribution) + list(APPROX_NAMESPACE_SIZE))):
        n = cov.namespace_distribution.get(ns, 0)
        size = APPROX_NAMESPACE_SIZE.get(ns, "?")
        lines.append(f"| {ns} | {n} | {size} |")
    if cov.blind_spots:
        lines += ["", "## Blind spots (0 coverage despite non-trivial corpus size)"]
        for ns in cov.blind_spots:
            lines.append(f"- **{ns}** (≈{APPROX_NAMESPACE_SIZE[ns]} chunks)")
    if cov.near_duplicate_pairs:
        lines += ["", "## Near-duplicate scenario pairs (Jaccard > 0.7)"]
        for a, b, j in sorted(cov.near_duplicate_pairs, key=lambda t: -t[2]):
            lines.append(f"- {a} ↔ {b}: Jaccard={j}")
    return "\n".join(lines) + "\n"
