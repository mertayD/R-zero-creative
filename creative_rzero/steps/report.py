"""steps/report.py — one W&B run per co-evolution run, holding every table
you need to judge that run's health.

Before this module, a single co-evolution run scattered its telemetry across
four kinds of W&B run: the verl training run (per-step reward metrics), a
throwaway `*_rollouts` run per phase (the rollout table), a `prompt-gen` run
(Phase B counters), and a `coevolve_summary` run that logged config and
immediately finished. Answering "is this run healthy?" meant opening all of
them and joining by name.

Everything now lands in the *summary* run instead, keyed by a deterministic
id (`{abbr}_{run_ts}`) so the CPU orchestrator and both GPU phase containers
all attach to the same run via `resume="allow"`. The phases are sequential,
so there's never more than one writer.

Tables logged (all rebuilt from the on-volume artifacts, so the newest
version of each is always complete):

  run_health              one row per (iteration, role) — the at-a-glance table
  {role}/rollouts         every rollout row, including raw_output/format_reason
  {role}/failure_breakdown  failure_reason + format_reason counts with shares
  {role}/step_health      per-global_step policy-collapse rate and flags
  prompt_gen/failures     Phase B failed attempts with full raw responses
  prompt_gen/breakdown    Phase B failure-reason counts

Reporting is telemetry: every entry point swallows its own exceptions, so a
W&B outage can never fail a training run that otherwise succeeded.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from creative_rzero.failure_reasons import INFRA_FAILURE_REASONS, FailureReason
from creative_rzero.paths import RunPaths

# W&B tables get sluggish well before they get rejected; cap rows rather than
# silently shipping a table nobody can open. Rollout logs are append-only, so
# the newest rows are the interesting ones when a run is long.
MAX_TABLE_ROWS = 2000


def wandb_project() -> str:
    """Read at call time, not import time — the env is assembled per container."""
    return os.environ.get("WANDB_PROJECT", "r-zero-creative")


def summary_run_id(paths: RunPaths) -> str:
    """Deterministic W&B run id for a co-evolution run.

    Derived from abbr + run_ts (not random, not stored in state) so any
    container in the run can `resume` the same summary run without passing an
    id across the Modal boundary.
    """
    return f"{paths.abbr}_{paths.run_ts}"


def _wandb_enabled() -> bool:
    if os.environ.get("WANDB_MODE", "online") == "disabled":
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


def _open_summary_run(paths: RunPaths):
    """Init-or-resume this run's summary W&B run. Returns None when W&B is
    unavailable/disabled, so callers can no-op."""
    if not _wandb_enabled():
        return None
    import wandb

    return wandb.init(
        project=wandb_project(),
        id=summary_run_id(paths),
        name=f"coevolve_{paths.abbr}_{paths.run_ts}",
        group=os.environ.get("WANDB_RUN_GROUP"),
        job_type="coevolve_summary",
        resume="allow",
        reinit=True,
    )


# ---------------------------------------------------------------------------
# Artifact readers
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> List[dict]:
    """Read a JSONL file, skipping blank and unparseable lines.

    Reward logs are appended by a training subprocess that can be killed
    mid-write, so a torn final line is expected rather than exceptional.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows: List[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def parse_log_name(path: Path, abbr: str, run_ts: str) -> Optional[Tuple[int, str]]:
    """`{abbr}_{run_ts}_iter{N}_{role}_v1.jsonl` -> (N, role).

    Strips the known prefix/suffix rather than splitting on "_", because
    `abbr` itself may contain underscores.
    """
    name = path.name
    prefix, suffix = f"{abbr}_{run_ts}_", "_v1.jsonl"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    middle = name[len(prefix): -len(suffix)]  # "iter3_challenger"
    iter_part, _, role = middle.partition("_")
    if not role or not iter_part.startswith("iter"):
        return None
    try:
        return int(iter_part[len("iter"):]), role
    except ValueError:
        return None


def find_rollout_logs(paths: RunPaths) -> List[Tuple[int, str, Path]]:
    """Every rollout log belonging to this run, as (iteration, role, path)."""
    log_dir = paths.storage / "reward_logs"
    if not log_dir.is_dir():
        return []
    found: List[Tuple[int, str, Path]] = []
    for p in log_dir.glob(f"{paths.abbr}_{paths.run_ts}_iter*_v1.jsonl"):
        parsed = parse_log_name(p, paths.abbr, paths.run_ts)
        if parsed is not None:
            found.append((parsed[0], parsed[1], p))
    return sorted(found)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _reward(row: dict) -> float:
    """Challenger rows carry `overall`; solver rows carry `rank_reward`."""
    value = row.get("overall", row.get("rank_reward", 0.0))
    return float(value) if isinstance(value, (int, float)) else 0.0


def rollout_health(rows: List[dict]) -> Dict[str, float]:
    """Scalar health summary for one phase's rollouts.

    `format_valid_rate` is only meaningful for challenger rows (solver rows
    have no format gate) and is omitted when the column is absent.
    """
    n = len(rows)
    if n == 0:
        return {"n_rollouts": 0}

    reasons = [str(r.get("failure_reason", FailureReason.OK.value)) for r in rows]
    n_ok = sum(1 for r in reasons if r == FailureReason.OK.value)
    n_infra = sum(1 for r in reasons if r in {m.value for m in INFRA_FAILURE_REASONS})

    health: Dict[str, float] = {
        "n_rollouts": n,
        "ok_rate": n_ok / n,
        "infra_failure_rate": n_infra / n,
        "policy_failure_rate": (n - n_ok - n_infra) / n,
        "mean_reward": sum(_reward(r) for r in rows) / n,
        "mean_accuracy": sum(float(r.get("accuracy", 0.0) or 0.0) for r in rows) / n,
    }
    if any("format_valid" in r for r in rows):
        health["format_valid_rate"] = sum(int(r.get("format_valid", 0)) for r in rows) / n
    return health


def top_reason(rows: List[dict], key: str = "failure_reason") -> str:
    """Most common non-ok value of `key` — the headline for a bad phase."""
    counts = Counter(
        str(r[key]) for r in rows
        if key in r and str(r[key]) != FailureReason.OK.value
    )
    return counts.most_common(1)[0][0] if counts else FailureReason.OK.value


def breakdown_rows(rows: List[dict], keys: Tuple[str, ...] = ("failure_reason", "format_reason")) -> List[dict]:
    """Long-format counts for every reason column present, so one table can
    hold both taxonomies: [{kind, value, count, share}, ...]."""
    out: List[dict] = []
    for key in keys:
        present = [str(r[key]) for r in rows if key in r]
        if not present:
            continue
        total = len(present)
        for value, count in Counter(present).most_common():
            out.append({"kind": key, "value": value, "count": count, "share": count / total})
    return out


# ---------------------------------------------------------------------------
# Reporting entry points
# ---------------------------------------------------------------------------

def _log_table(run, key: str, rows: List[dict], *, cap: int = MAX_TABLE_ROWS) -> None:
    if not rows:
        return
    import pandas as pd
    import wandb

    capped = rows[-cap:]
    run.log({key: wandb.Table(dataframe=pd.DataFrame(capped))})


def build_run_health(paths: RunPaths) -> List[dict]:
    """The at-a-glance table: one row per (iteration, role) across the whole
    run, rebuilt from every rollout log on the volume so the latest logged
    version is always the complete picture."""
    health_rows: List[dict] = []
    for iteration, role, log_path in find_rollout_logs(paths):
        rows = read_jsonl(log_path)
        entry: Dict[str, Any] = {"iteration": iteration, "role": role}
        entry.update(rollout_health(rows))
        entry["top_failure_reason"] = top_reason(rows)
        if any("format_reason" in r for r in rows):
            entry["top_format_reason"] = top_reason(rows, "format_reason")
        health_rows.append(entry)
    return health_rows


def report_phase(paths: RunPaths, role: str, *, max_steps: Optional[int] = None) -> None:
    """Log one finished phase's tables into the run's summary W&B run.

    Reads only on-volume artifacts, so it can run in any container and is
    safe to re-run (tables are rebuilt, never appended to). `max_steps`
    enables the per-step health table; omit it to skip that table.
    """
    try:
        run = _open_summary_run(paths)
        if run is None:
            print("[report] W&B disabled or unauthenticated — skipping run report")
            return

        rows = read_jsonl(paths.rollout_log(role))
        _log_table(run, f"{role}/rollouts", rows)
        _log_table(run, f"{role}/failure_breakdown", breakdown_rows(rows))

        if max_steps is not None:
            from creative_rzero.steps.merge_ckpt import compute_step_health

            health = compute_step_health(paths.rollout_log(role))
            _log_table(run, f"{role}/step_health", [
                {
                    "global_step": h.step,
                    "policy_collapse_rate": h.policy_collapse_rate,
                    "n_groups": h.n_groups,
                    "hard_flag": h.hard_flag,
                    "soft_flag": h.soft_flag,
                    "flagged": h.flagged,
                }
                for h in sorted(health.values(), key=lambda h: h.step)
            ])

        _log_table(run, "run_health", build_run_health(paths))

        # Summary scalars surface as sortable columns in the W&B runs list,
        # which is how you compare health *across* runs rather than within one.
        for metric, value in rollout_health(rows).items():
            run.summary[f"{role}/iter{paths.iteration}/{metric}"] = value
        run.summary[f"{role}/iter{paths.iteration}/top_failure_reason"] = top_reason(rows)

        print(f"[report] {role} iter{paths.iteration}: logged {len(rows)} rollouts to W&B run {summary_run_id(paths)}")
    except Exception as e:  # telemetry must never kill a run
        print(f"[report] phase report failed ({role}): {e}")


def report_generation(paths: RunPaths, prompts_json: str | Path) -> None:
    """Log Phase B prompt-generation artifacts (the batch's generation_log and
    the failed-attempt sidecar) into the same summary run."""
    try:
        run = _open_summary_run(paths)
        if run is None:
            return

        prompts_path = Path(prompts_json)
        if prompts_path.exists():
            batch = json.loads(prompts_path.read_text())
            log = batch.get("generation_log", {})
            counts = log.get("failure_reason_counts", {}) or {}
            total_failed = sum(counts.values())
            _log_table(run, "prompt_gen/breakdown", [
                {"value": reason, "count": count,
                 "share": (count / total_failed) if total_failed else 0.0}
                for reason, count in sorted(counts.items(), key=lambda kv: -kv[1])
            ])
            run.summary[f"prompt_gen/iter{paths.iteration}/n_prompts"] = len(batch.get("prompts", []))
            for metric in ("format_validation_failures", "language_filter_failures"):
                if metric in log:
                    run.summary[f"prompt_gen/iter{paths.iteration}/{metric}"] = log[metric]

        failures = read_jsonl(prompts_path.with_suffix(".failures.jsonl"))
        _log_table(run, "prompt_gen/failures", failures)

        print(f"[report] prompt-gen iter{paths.iteration}: logged {len(failures)} failed attempts to W&B")
    except Exception as e:
        print(f"[report] generation report failed: {e}")
