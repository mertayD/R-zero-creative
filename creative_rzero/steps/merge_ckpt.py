"""steps/merge_ckpt.py — merges a VERL FSDP checkpoint into a HuggingFace
directory, with T1.11's health-based step selection built in.

The old bash scripts always merged `global_step_{max_steps - 1}` (the last
step trained) via `scripts/model_merger.py`, regardless of how that step's
rollouts actually scored. REFACTOR_PLAN.md's T1.11 traced a real run where
the solver's iter-1 checkpoint was merged from deep inside a step-19..32
collapse region — feeding a degenerate policy into the next co-evolution
round. This module reads the same rollout JSONL log the reward callers
already write (T1.4's `failure_reason` taxonomy) to pick the last *healthy*
step instead.

A flagged step does NOT block the merge by default — the thresholds below
haven't been validated against a real run yet, so refusing outright would
just be a confident guess. Instead, `merge_checkpoint` always logs the
selected step's health (rate, hard/soft flag, group count) so it's visible
in the run log regardless of outcome, and loudly warns when it's flagged.
Pass `strict=True` to opt into the harder behavior (raise
`CollapsedCheckpointError` instead of warning) once the thresholds earn
that trust.

Collapse detection (per the current T2.7 spec):

  - Every non-"ok" sample is split into `infra` (`judge_api_error`,
    `judge_rate_limit` — harmless to the checkpoint itself: zero reward is
    zero GRPO advantage either way, it says nothing about the policy) vs.
    `policy` (everything else non-"ok" by exclusion, plus a solver row
    flagged `truncated` or a group flagged `is_low_quality` even when its
    failure_reason is "ok" — i.e. scored, but scored as garbage). Only
    policy-attributable failure can flag a step; infra noise is invisible to
    the detector.
  - A rollout *group* (the solver's G samples per prompt_id; a lone row for
    the challenger, which logs no sub-grouping) is "policy-collapsed" if
    every sample in it is a policy failure.
  - `hard_flag`: policy-collapse rate > 50% at that step alone.
  - `soft_flag`: 3 consecutive present steps with a rate > 2x the run's
    baseline (mean rate over the earliest steps logged, up to 3) and
    monotonically increasing across those three — catches the ramp before a
    cliff (the plan's iter-1 trace put this at step ~17-18, well before the
    step-19 collapse).
  - `healthy`: unflagged (neither hard nor soft) and rate < 10%.
  - Selection: the latest healthy step; if none is healthy, the least-bad
    step available (non-flagged preferred over flagged, then lowest rate,
    then most recent).

Known gap vs. the full T2.7 spec: this is retrospective only — it reads the
finished run's JSONL after the fact. `trainer.save_limit` in
configs/base.yaml is set to -1 (keep every checkpoint) specifically so a
step this picks can never have already been pruned off disk by the time
`merge_checkpoint` runs; see that key's comment for the tradeoff (disk vs.
engineering complexity). The plan's preferred alternative — an *online*
monitor that kills the trainer subprocess mid-run the moment the soft flag
trips, so no compute/judge spend goes to steps beyond the ramp — is not
implemented.
TODO(dayanc): consider adding online monitoring for collapse (would live in
steps/train_verl.py's launch_training, which already streams the trainer
subprocess's output live and could tail the rollout JSONL alongside it).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from creative_rzero.failure_reasons import INFRA_FAILURE_REASONS, FailureReason
from creative_rzero.paths import RunPaths

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_MERGER_SCRIPT = REPO_ROOT / "scripts" / "model_merger.py"

HARD_FLAG_RATE = 0.50  # policy-collapse rate above this at any single step is an outright flag
HEALTHY_RATE = 0.10  # policy-collapse rate must be below this (and unflagged) to count healthy
BASELINE_STEPS = 3  # earliest N logged steps define the run's baseline rate
SOFT_FLAG_RUN_LENGTH = 3  # consecutive worsening steps required to trip the soft (ramp) flag
SOFT_FLAG_MULTIPLIER = 2.0  # a step must exceed baseline * this to count toward the ramp


class CollapsedCheckpointError(RuntimeError):
    """Raised by `merge_checkpoint` when the selected/requested step is
    hard- or soft-flagged and `strict=True` was passed."""


class MergeFailedError(RuntimeError):
    """Raised when scripts/model_merger.py exits nonzero."""


@dataclass
class StepHealth:
    step: int
    policy_collapse_rate: float
    n_groups: int
    hard_flag: bool = False
    soft_flag: bool = False

    @property
    def flagged(self) -> bool:
        return self.hard_flag or self.soft_flag


def _group_key(row: dict) -> str:
    """Solver rows carry `prompt_id` (G samples share one); challenger rows
    don't (one row per generated prompt, no further grouping) — falling back
    to `rollout_idx` makes every challenger row its own group of one."""
    if "prompt_id" in row:
        return row["prompt_id"]
    return f"row{row.get('rollout_idx', id(row))}"


def _log_step_to_global_step(log_step: int) -> int:
    """Both reward callers log a 1-indexed monotonic call counter (one
    increment per `compute_score` call), not VERL's own 0-indexed
    `global_step_N`. Assuming one `compute_score` call per training step
    (true for both creative callers today), log step N is global_step N-1."""
    return log_step - 1


def _is_policy_failure(row: dict) -> bool:
    """A sample counts against a step's health if its failure isn't
    infra-attributable, or if it scored but scored as garbage: a solver
    sample flagged `truncated`, or one in a group flagged `is_low_quality`
    (T1.6's all-scores-at-floor gate) — both log `failure_reason: "ok"`
    despite being exactly the kind of policy degeneration this is meant to
    catch. Unknown/future failure_reason values default to policy (the safer
    failure mode — silently ignoring a new reason would mask real collapse)."""
    reason = row.get("failure_reason", FailureReason.OK)
    if reason != FailureReason.OK and reason not in INFRA_FAILURE_REASONS:
        return True
    if row.get("truncated"):
        return True
    if row.get("is_low_quality"):
        return True
    return False


def compute_step_health(rollout_log: str | Path) -> dict[int, StepHealth]:
    """Read a role's rollout JSONL log and compute, per VERL global_step,
    the policy-collapse rate plus the hard/soft flags derived from it.

    Returns `{}` if the log doesn't exist — dry runs and mock-judge profiles
    have no rollout history, and "no data" must not be conflated with
    "collapsed"; callers fall back to the old `max_steps - 1` default in
    that case instead.
    """
    rollout_log = Path(rollout_log)
    if not rollout_log.exists():
        return {}

    # global_step -> group_key -> [row, ...]
    groups_by_step: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with open(rollout_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            global_step = _log_step_to_global_step(row["step"])
            groups_by_step[global_step][_group_key(row)].append(row)

    rates: dict[int, tuple[float, int]] = {}
    for global_step, groups in groups_by_step.items():
        n_groups = len(groups)
        n_collapsed = sum(1 for rows in groups.values() if all(_is_policy_failure(r) for r in rows))
        rates[global_step] = ((n_collapsed / n_groups) if n_groups else 0.0, n_groups)

    if not rates:
        return {}

    sorted_steps = sorted(rates)
    baseline_steps = sorted_steps[:BASELINE_STEPS]
    baseline = sum(rates[s][0] for s in baseline_steps) / len(baseline_steps)

    health: dict[int, StepHealth] = {
        s: StepHealth(step=s, policy_collapse_rate=rate, n_groups=n) for s, (rate, n) in rates.items()
    }

    for i, step in enumerate(sorted_steps):
        if i < SOFT_FLAG_RUN_LENGTH - 1:
            continue
        window = sorted_steps[i - (SOFT_FLAG_RUN_LENGTH - 1): i + 1]
        # Only a run of *consecutive* global_steps counts as a ramp — gaps in
        # the log (e.g. a step with zero groups) break the window.
        if window[-1] - window[0] != SOFT_FLAG_RUN_LENGTH - 1:
            continue
        window_rates = [rates[s][0] for s in window]
        monotonically_worsening = all(a < b for a, b in zip(window_rates, window_rates[1:]))
        above_baseline = all(r > baseline * SOFT_FLAG_MULTIPLIER for r in window_rates)
        if monotonically_worsening and above_baseline:
            health[step].soft_flag = True

    for h in health.values():
        h.hard_flag = h.policy_collapse_rate > HARD_FLAG_RATE

    return health


def select_checkpoint_step(
    rollout_log: str | Path,
    max_steps: int,
) -> tuple[int, Optional[StepHealth]]:
    """Pick the step to merge: the latest global_step (0-indexed, < max_steps)
    that's unflagged with a policy-collapse rate below HEALTHY_RATE. If none
    qualifies, fall back to the least-bad step available (unflagged preferred
    over flagged, then lowest rate, then most recent). If the log has no rows
    for any step in range at all, fall back to `max_steps - 1` — the old
    scripts' blind convention — with `health=None` since there's no signal.
    """
    health = compute_step_health(rollout_log)
    candidates = [health[s] for s in range(max_steps) if s in health]
    if not candidates:
        return max_steps - 1, None

    healthy = [h for h in candidates if not h.flagged and h.policy_collapse_rate < HEALTHY_RATE]
    if healthy:
        best = max(healthy, key=lambda h: h.step)
    else:
        best = min(candidates, key=lambda h: (h.flagged, h.policy_collapse_rate, -h.step))
    return best.step, best


def merge_checkpoint(
    paths: RunPaths,
    role: str,
    *,
    step: Optional[int] = None,
    max_steps: Optional[int] = None,
    strict: bool = False,
    run: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run,
) -> Path:
    """Merge one VERL FSDP checkpoint into a HuggingFace directory.

    If `step` is omitted, it's picked via `select_checkpoint_step` over
    `paths.rollout_log(role)` (requires `max_steps`). The selected step's
    health is always logged — printed plainly when healthy, as a loud
    warning when hard- or soft-flagged — so collapse is visible in every run
    log (REFACTOR_PLAN.md T1.11) without yet trusting the thresholds enough
    to gate on them. Pass `strict=True` to raise `CollapsedCheckpointError`
    on a flagged step instead of warning and proceeding.

    `run` is injectable so callers/tests can substitute a fake instead of
    actually invoking `scripts/model_merger.py`. On success, writes the
    merged path to `paths.last_ckpt_file(role)` — the sidecar the
    orchestrator reads back instead of reconstructing SAVE_NAME/global_step
    itself (REFACTOR_PLAN.md class F) — and returns the merged HuggingFace
    directory path.
    """
    if step is None:
        if max_steps is None:
            raise ValueError("merge_checkpoint requires max_steps when step is not given")
        step, health = select_checkpoint_step(paths.rollout_log(role), max_steps)
    else:
        health = compute_step_health(paths.rollout_log(role)).get(step)

    if health is None:
        print(f"[merge_ckpt] {role} global_step_{step}: no rollout-log health data available")
    else:
        print(
            f"[merge_ckpt] {role} global_step_{step}: policy_collapse_rate="
            f"{health.policy_collapse_rate:.1%} hard_flag={health.hard_flag} "
            f"soft_flag={health.soft_flag} n_groups={health.n_groups}"
        )
        if health.flagged:
            flag_kind = "hard-flagged" if health.hard_flag else "soft-flagged (ramp)"
            print(
                f"[merge_ckpt] WARNING: {role} global_step_{step} is {flag_kind} "
                f"({health.policy_collapse_rate:.0%} policy-collapse rate) — merging anyway "
                "(pass strict=True to refuse instead)."
            )
            if strict:
                raise CollapsedCheckpointError(
                    f"{role} global_step_{step} is {flag_kind}: {health.policy_collapse_rate:.0%} "
                    "policy-collapse rate — refusing to merge a collapsed checkpoint under strict=True."
                )

    local_dir = paths.checkpoint_step_dir(role, step)
    cmd = [sys.executable, str(MODEL_MERGER_SCRIPT), "--local_dir", str(local_dir)]
    result = run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise MergeFailedError(
            f"model_merger.py exited {result.returncode} for {role} global_step_{step}:\n{result.stderr}"
        )

    merged_path = paths.merged_checkpoint(role, step)
    last_ckpt_file = paths.last_ckpt_file(role)
    last_ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt_file.write_text(str(merged_path))

    return merged_path
