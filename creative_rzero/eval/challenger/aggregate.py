"""creative_rzero/eval/challenger/aggregate.py — stratified rollup of a
run_eval.py per-row scoring pass into overall / per-domain / per-subdomain
summary stats.

Pattern reference (not shared code — different schema):
evaluation/writing_bench/calculate_scores.py, which aggregates solver-side
per-criterion judge scores the same "roll up from a per-row JSONL" way.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _group_stats(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    format_valid_rows = [r for r in rows if r["format_valid"]]

    failure_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if not r["format_valid"]:
            failure_counts[r["format_failure_reason"]] += 1

    judged_rows = [r for r in format_valid_rows if r["domain_adherence"] is not None]
    dup_rows = [r for r in format_valid_rows if r.get("near_duplicate") is not None]
    query_lens = [r["query_len"] for r in format_valid_rows]

    return {
        "n": n,
        "format_pass_rate": len(format_valid_rows) / n if n else None,
        "format_failure_reason_counts": dict(failure_counts),
        "domain_adherence_mean": _mean_or_none([r["domain_adherence"] for r in judged_rows]),
        "guidance_adherence_mean": _mean_or_none([r["guidance_adherence"] for r in judged_rows]),
        "criteria_quality_mean": _mean_or_none([r["criteria_quality"] for r in judged_rows]),
        "duplicate_rate": (
            sum(1 for r in dup_rows if r["near_duplicate"]) / len(dup_rows) if dup_rows else None
        ),
        "query_len_mean": _mean_or_none(query_lens),
        "query_len_stddev": pstdev(query_lens) if len(query_lens) > 1 else None,
        "criteria_len_mean": _mean_or_none([r["criteria_len"] for r in format_valid_rows]),
    }


def aggregate_rows(rows: list[dict]) -> dict[str, Any]:
    """Stratify `rows` (run_eval.py's per-row records — see its module
    docstring for the schema) into overall, per-domain, and per-subdomain
    summaries. `by_subdomain` is keyed `"<domain>::<subdomain>"` since
    subdomain names are not unique across domains."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    by_subdomain: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
        by_subdomain[f"{r['domain']}::{r['subdomain']}"].append(r)

    return {
        "overall": _group_stats(rows),
        "by_domain": {k: _group_stats(v) for k, v in sorted(by_domain.items())},
        "by_subdomain": {k: _group_stats(v) for k, v in sorted(by_subdomain.items())},
    }
