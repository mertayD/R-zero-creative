import json

import pytest

from creative_rzero.failure_reasons import FailureReason
from creative_rzero.paths import RunPaths
from creative_rzero.steps.merge_ckpt import (
    CollapsedCheckpointError,
    MergeFailedError,
    StepHealth,
    compute_step_health,
    merge_checkpoint,
    select_checkpoint_step,
)


@pytest.fixture
def paths(tmp_path) -> RunPaths:
    return RunPaths(tmp_path / "storage", "example", "20260805_120000", iteration=1)


def _write_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _solver_row(step, prompt_id, reason=FailureReason.OK, truncated=False, is_low_quality=False):
    return {
        "step": step,
        "prompt_id": prompt_id,
        "failure_reason": reason.value,
        "truncated": truncated,
        "is_low_quality": is_low_quality,
    }


def _challenger_row(step, rollout_idx, reason=FailureReason.OK):
    return {"step": step, "rollout_idx": rollout_idx, "failure_reason": reason.value}


def _healthy_step_rows(log_step, n_groups=4):
    """A clean step: n_groups solver groups, each with one ok sample."""
    return [_solver_row(log_step, f"p{log_step}_{i}") for i in range(n_groups)]


def _collapsed_step_rows(log_step, n_groups=4):
    """A fully policy-collapsed step: every group entirely empty_answer'd."""
    return [
        _solver_row(log_step, f"p{log_step}_{i}", reason=FailureReason.EMPTY_ANSWER)
        for i in range(n_groups)
    ]


# ---------------------------------------------------------------------------
# compute_step_health — policy/infra split
# ---------------------------------------------------------------------------


def test_compute_step_health_missing_log_returns_empty(paths):
    assert compute_step_health(paths.rollout_log("solver")) == {}


def test_compute_step_health_groups_solver_rows_by_prompt_id(tmp_path):
    log_path = tmp_path / "log.jsonl"
    # log step 1 -> global_step 0: one group all-failed (policy), one all-ok
    _write_log(log_path, [
        _solver_row(1, "p1", reason=FailureReason.EMPTY_ANSWER),
        _solver_row(1, "p1", reason=FailureReason.EMPTY_ANSWER),
        _solver_row(1, "p2", reason=FailureReason.OK),
        _solver_row(1, "p2", reason=FailureReason.OK),
    ])

    health = compute_step_health(log_path)

    assert health[0].policy_collapse_rate == 0.5
    assert health[0].n_groups == 2


def test_compute_step_health_infra_failures_do_not_count_as_policy_collapse(tmp_path):
    log_path = tmp_path / "log.jsonl"
    # every sample failed, but only on infra reasons -> not policy-collapsed
    _write_log(log_path, [
        _solver_row(1, "p1", reason=FailureReason.JUDGE_API_ERROR),
        _solver_row(1, "p1", reason=FailureReason.JUDGE_RATE_LIMIT),
        _solver_row(1, "p2", reason=FailureReason.OK),
        _solver_row(1, "p2", reason=FailureReason.OK),
    ])

    health = compute_step_health(log_path)

    assert health[0].policy_collapse_rate == 0.0


def test_compute_step_health_truncated_flag_counts_even_when_ok(tmp_path):
    log_path = tmp_path / "log.jsonl"
    # failure_reason "ok" but both samples truncated -> group counts as policy-collapsed
    _write_log(log_path, [
        _solver_row(1, "p1", reason=FailureReason.OK, truncated=True),
        _solver_row(1, "p1", reason=FailureReason.OK, truncated=True),
    ])

    health = compute_step_health(log_path)

    assert health[0].policy_collapse_rate == 1.0


def test_compute_step_health_is_low_quality_flag_counts_even_when_ok(tmp_path):
    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, [
        _solver_row(1, "p1", reason=FailureReason.OK, is_low_quality=True),
        _solver_row(1, "p1", reason=FailureReason.OK, is_low_quality=True),
    ])

    health = compute_step_health(log_path)

    assert health[0].policy_collapse_rate == 1.0


def test_compute_step_health_partial_group_failure_not_collapsed(tmp_path):
    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, [
        _solver_row(1, "p1", reason=FailureReason.EMPTY_ANSWER),
        _solver_row(1, "p1", reason=FailureReason.OK),
    ])

    health = compute_step_health(log_path)

    assert health[0].policy_collapse_rate == 0.0


def test_compute_step_health_challenger_rows_have_no_subgrouping(tmp_path):
    log_path = tmp_path / "log.jsonl"
    # 3/4 rows failed -> 75% policy-collapse rate, strictly above the 50% hard-flag threshold
    _write_log(log_path, [
        _challenger_row(1, 0, reason=FailureReason.CHALLENGER_FORMAT_INVALID),
        _challenger_row(1, 1, reason=FailureReason.CHALLENGER_FORMAT_INVALID),
        _challenger_row(1, 2, reason=FailureReason.CHALLENGER_FORMAT_INVALID),
        _challenger_row(1, 3, reason=FailureReason.OK),
    ])

    health = compute_step_health(log_path)

    assert health[0] == StepHealth(step=0, policy_collapse_rate=0.75, n_groups=4, hard_flag=True)


def test_compute_step_health_unknown_reason_defaults_to_policy(tmp_path):
    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, [{"step": 1, "prompt_id": "p1", "failure_reason": "some_future_reason"}])

    health = compute_step_health(log_path)

    assert health[0].policy_collapse_rate == 1.0


# ---------------------------------------------------------------------------
# compute_step_health — hard / soft flags
# ---------------------------------------------------------------------------


def test_hard_flag_trips_above_50_percent_at_a_single_step(tmp_path):
    log_path = tmp_path / "log.jsonl"
    rows = _collapsed_step_rows(1, n_groups=3) + [_solver_row(1, "ok_group", reason=FailureReason.OK)]
    # 3/4 groups collapsed = 75% > 50%
    _write_log(log_path, rows)

    health = compute_step_health(log_path)

    assert health[0].hard_flag is True
    assert health[0].flagged is True


def test_soft_flag_trips_on_monotonic_ramp_above_baseline(tmp_path):
    log_path = tmp_path / "log.jsonl"
    rows = []
    # baseline steps (global_step 0,1,2): clean
    for gs in range(3):
        rows += _healthy_step_rows(gs + 1)
    # ramp: global_step 3,4,5 monotonically worsening, each > 2x baseline (0.0)
    rows += [_solver_row(4, "p_a", reason=FailureReason.EMPTY_ANSWER)] + _healthy_step_rows(4, n_groups=9)
    rows += (
        [_solver_row(5, "p_a", reason=FailureReason.EMPTY_ANSWER),
         _solver_row(5, "p_b", reason=FailureReason.EMPTY_ANSWER)]
        + _healthy_step_rows(5, n_groups=8)
    )
    rows += (
        [_solver_row(6, f"p_{i}", reason=FailureReason.EMPTY_ANSWER) for i in range(3)]
        + _healthy_step_rows(6, n_groups=7)
    )
    _write_log(log_path, rows)

    health = compute_step_health(log_path)

    assert health[3].policy_collapse_rate == pytest.approx(0.1)
    assert health[4].policy_collapse_rate == pytest.approx(0.2)
    assert health[5].policy_collapse_rate == pytest.approx(0.3)
    assert health[5].soft_flag is True
    assert health[5].flagged is True
    # step 4 is part of the ramp window ending at 5, not itself a 3-run end
    assert health[3].soft_flag is False


def test_soft_flag_requires_consecutive_global_steps(tmp_path):
    log_path = tmp_path / "log.jsonl"
    rows = []
    for gs in range(3):
        rows += _healthy_step_rows(gs + 1)
    # gap: global_step 3 missing, then two worsening steps at 4, 5 -- window
    # [3,4,5] isn't fully present so the 3-run-consecutive check can't fire
    rows += [_solver_row(5, "p_a", reason=FailureReason.EMPTY_ANSWER)] + _healthy_step_rows(5, n_groups=9)
    rows += (
        [_solver_row(6, "p_a", reason=FailureReason.EMPTY_ANSWER),
         _solver_row(6, "p_b", reason=FailureReason.EMPTY_ANSWER)]
        + _healthy_step_rows(6, n_groups=8)
    )
    _write_log(log_path, rows)

    health = compute_step_health(log_path)

    assert health[5].soft_flag is False


# ---------------------------------------------------------------------------
# select_checkpoint_step
# ---------------------------------------------------------------------------


def test_select_checkpoint_step_no_log_falls_back_to_max_steps_minus_one(paths):
    step, health = select_checkpoint_step(paths.rollout_log("solver"), max_steps=6)

    assert step == 5
    assert health is None


def test_select_checkpoint_step_picks_latest_healthy_step(tmp_path):
    log_path = tmp_path / "log.jsonl"
    rows = _healthy_step_rows(1) + _healthy_step_rows(2) + _collapsed_step_rows(3)
    _write_log(log_path, rows)

    step, health = select_checkpoint_step(log_path, max_steps=3)

    assert step == 1
    assert health.policy_collapse_rate == 0.0


def test_select_checkpoint_step_falls_back_to_least_bad_when_none_healthy(tmp_path):
    log_path = tmp_path / "log.jsonl"
    # global_step 0: 80% collapsed; global_step 1: 60% collapsed (least bad, more recent)
    rows = (
        [_solver_row(1, "p0", reason=FailureReason.OK)]
        + [_solver_row(1, f"p{i}", reason=FailureReason.EMPTY_ANSWER) for i in range(1, 5)]
        + [_solver_row(2, "p0", reason=FailureReason.OK), _solver_row(2, "p1", reason=FailureReason.OK)]
        + [_solver_row(2, f"p{i}", reason=FailureReason.EMPTY_ANSWER) for i in range(2, 5)]
    )
    _write_log(log_path, rows)

    step, health = select_checkpoint_step(log_path, max_steps=2)

    assert step == 1
    assert health.policy_collapse_rate == pytest.approx(0.6)


def test_select_checkpoint_step_only_considers_steps_below_max_steps(tmp_path):
    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, _healthy_step_rows(6))  # global_step 5, out of range for max_steps=3

    step, health = select_checkpoint_step(log_path, max_steps=3)

    assert step == 2
    assert health is None


# ---------------------------------------------------------------------------
# merge_checkpoint
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


def test_merge_checkpoint_happy_path_writes_last_ckpt_file(paths):
    _write_log(paths.rollout_log("solver"), _healthy_step_rows(1))
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return _FakeResult(returncode=0)

    merged = merge_checkpoint(paths, "solver", max_steps=1, run=fake_run)

    assert merged == paths.merged_checkpoint("solver", 0)
    assert calls["cmd"][-2:] == ["--local_dir", str(paths.checkpoint_step_dir("solver", 0))]
    assert paths.last_ckpt_file("solver").read_text() == str(merged)


def test_merge_checkpoint_merges_flagged_step_by_default_but_warns(paths, capsys):
    _write_log(paths.rollout_log("solver"), _collapsed_step_rows(1))
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["ran"] = True
        return _FakeResult(returncode=0)

    merged = merge_checkpoint(paths, "solver", max_steps=1, run=fake_run)

    assert calls["ran"] is True
    assert merged == paths.merged_checkpoint("solver", 0)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "global_step_0" in out
    assert "hard-flagged" in out


def test_merge_checkpoint_logs_health_even_when_not_flagged(paths, capsys):
    _write_log(paths.rollout_log("solver"), _healthy_step_rows(1))

    merge_checkpoint(paths, "solver", max_steps=1, run=lambda *a, **k: _FakeResult(returncode=0))

    out = capsys.readouterr().out
    assert "global_step_0" in out
    assert "policy_collapse_rate=0.0%" in out
    assert "WARNING" not in out


def test_merge_checkpoint_strict_raises_on_flagged_step(paths):
    _write_log(paths.rollout_log("solver"), _collapsed_step_rows(1))

    with pytest.raises(CollapsedCheckpointError, match="global_step_0"):
        merge_checkpoint(paths, "solver", max_steps=1, strict=True, run=lambda *a, **k: _FakeResult())


def test_merge_checkpoint_explicit_step_still_checked_under_strict(paths):
    rows = _healthy_step_rows(1) + _collapsed_step_rows(2)
    _write_log(paths.rollout_log("solver"), rows)

    with pytest.raises(CollapsedCheckpointError):
        merge_checkpoint(paths, "solver", step=1, strict=True, run=lambda *a, **k: _FakeResult())

    # step 0 is healthy -> no refusal even under strict
    merged = merge_checkpoint(
        paths, "solver", step=0, strict=True, run=lambda *a, **k: _FakeResult(returncode=0)
    )
    assert merged == paths.merged_checkpoint("solver", 0)


def test_merge_checkpoint_raises_on_nonzero_exit(paths):
    def fake_run(cmd, **kwargs):
        return _FakeResult(returncode=1, stderr="boom")

    with pytest.raises(MergeFailedError, match="boom"):
        merge_checkpoint(paths, "solver", step=0, run=fake_run)


def test_merge_checkpoint_requires_max_steps_when_step_omitted(paths):
    with pytest.raises(ValueError, match="max_steps"):
        merge_checkpoint(paths, "solver", run=lambda *a, **k: _FakeResult())


# ---------------------------------------------------------------------------
# End-to-end shape check against the plan's iter-1 trace: a baseline, a ramp
# starting ~step 17-18, and a cliff at 19+ -- the detector must flag no later
# than step 18 and never select a step >= 19. (Synthetic: the real iter-1
# JSONL lives on the training volume, not in this repo.)
# ---------------------------------------------------------------------------


def test_synthetic_iter1_shaped_run_flags_before_the_cliff(tmp_path):
    log_path = tmp_path / "log.jsonl"
    rows = []
    n_groups = 20
    for gs in range(32):
        log_step = gs + 1
        if gs < 17:
            n_bad = 0  # clean baseline
        elif gs < 19:
            n_bad = 2 + (gs - 17) * 3  # ramp: steps 17, 18 -> 2, 5 bad groups
        else:
            n_bad = 18  # cliff: steps 19+ collapse (90%)
        rows += [
            _solver_row(log_step, f"p{gs}_{i}", reason=FailureReason.EMPTY_ANSWER) for i in range(n_bad)
        ]
        rows += [
            _solver_row(log_step, f"p{gs}_{i}", reason=FailureReason.OK) for i in range(n_bad, n_groups)
        ]
    _write_log(log_path, rows)

    step, health = select_checkpoint_step(log_path, max_steps=32)

    assert step <= 18
    assert not health.flagged

    health_by_step = compute_step_health(log_path)
    for gs in range(19, 32):
        assert health_by_step[gs].flagged, f"global_step {gs} (post-cliff) should be flagged"
