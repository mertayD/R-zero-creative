import json
from pathlib import Path

import pytest

from creative_rzero.paths import RunPaths
from creative_rzero.steps.report import (
    breakdown_rows,
    build_run_health,
    find_rollout_logs,
    parse_log_name,
    read_jsonl,
    report_generation,
    report_phase,
    rollout_health,
    summary_run_id,
    top_reason,
)

ABBR = "example"
RUN_TS = "20260805_120000"


@pytest.fixture
def paths(tmp_path) -> RunPaths:
    return RunPaths(tmp_path, ABBR, RUN_TS, iteration=1)


def _write_log(paths: RunPaths, role: str, rows: list[dict], iteration: int = 1) -> Path:
    p = RunPaths(paths.storage, paths.abbr, paths.run_ts, iteration).rollout_log(role)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _challenger_row(idx: int, reason: str = "ok", fmt: int = 1, overall: float = 0.5) -> dict:
    return {
        "step": 1, "rollout_idx": idx, "format_valid": fmt, "format_reason": "ok",
        "failure_reason": reason, "overall": overall, "accuracy": 0.4, "format": 1.0,
    }


# =============================================================================
# Artifact readers
# =============================================================================

def test_summary_run_id_is_deterministic_from_abbr_and_run_ts(paths):
    # Every container in a run must derive the same id without coordination.
    assert summary_run_id(paths) == f"{ABBR}_{RUN_TS}"
    assert summary_run_id(RunPaths("/other", ABBR, RUN_TS, iteration=9)) == summary_run_id(paths)


def test_read_jsonl_skips_blank_and_torn_lines(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3\n')  # last line truncated mid-write

    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_parse_log_name_extracts_iteration_and_role():
    p = Path(f"{ABBR}_{RUN_TS}_iter3_challenger_v1.jsonl")
    assert parse_log_name(p, ABBR, RUN_TS) == (3, "challenger")


def test_parse_log_name_handles_abbr_containing_underscores():
    abbr = "my_run_name"
    p = Path(f"{abbr}_{RUN_TS}_iter2_solver_v1.jsonl")
    assert parse_log_name(p, abbr, RUN_TS) == (2, "solver")


def test_parse_log_name_rejects_other_runs():
    p = Path(f"other-abbr_{RUN_TS}_iter1_solver_v1.jsonl")
    assert parse_log_name(p, ABBR, RUN_TS) is None


def test_find_rollout_logs_returns_only_this_runs_logs_sorted(paths):
    _write_log(paths, "challenger", [_challenger_row(0)], iteration=1)
    _write_log(paths, "solver", [_challenger_row(0)], iteration=1)
    _write_log(paths, "challenger", [_challenger_row(0)], iteration=2)
    # A different run sharing the volume must not leak into this run's report.
    other = RunPaths(paths.storage, ABBR, "20990101_000000", 1).rollout_log("challenger")
    other.write_text("{}\n")

    found = find_rollout_logs(paths)

    assert [(i, role) for i, role, _ in found] == [
        (1, "challenger"), (1, "solver"), (2, "challenger"),
    ]


# =============================================================================
# Aggregation
# =============================================================================

def test_rollout_health_splits_infra_from_policy_failures():
    rows = [
        _challenger_row(0, "ok"),
        _challenger_row(1, "judge_api_error", overall=0.0),   # infra
        _challenger_row(2, "truncated", fmt=0, overall=0.0),  # policy
        _challenger_row(3, "judge_rate_limit", overall=0.0),  # infra
    ]

    health = rollout_health(rows)

    assert health["n_rollouts"] == 4
    assert health["ok_rate"] == 0.25
    assert health["infra_failure_rate"] == 0.5
    assert health["policy_failure_rate"] == 0.25
    assert health["format_valid_rate"] == 0.75


def test_rollout_health_empty_log():
    assert rollout_health([]) == {"n_rollouts": 0}


def test_rollout_health_reads_solver_rank_reward_as_the_reward():
    # Solver rows name their reward `rank_reward`; challenger rows use `overall`.
    rows = [{"failure_reason": "ok", "rank_reward": 1.0}, {"failure_reason": "ok", "rank_reward": 0.0}]

    assert rollout_health(rows)["mean_reward"] == 0.5


def test_rollout_health_omits_format_rate_for_solver_rows():
    assert "format_valid_rate" not in rollout_health([{"failure_reason": "ok", "rank_reward": 1.0}])


def test_top_reason_ignores_ok_and_picks_the_most_common():
    rows = [
        _challenger_row(0, "ok"), _challenger_row(1, "ok"),
        _challenger_row(2, "judge_api_error"), _challenger_row(3, "judge_api_error"),
        _challenger_row(4, "truncated"),
    ]

    assert top_reason(rows) == "judge_api_error"


def test_top_reason_is_ok_when_nothing_failed():
    assert top_reason([_challenger_row(0, "ok")]) == "ok"


def test_breakdown_rows_covers_both_taxonomies_with_shares():
    rows = [
        {"failure_reason": "ok", "format_reason": "ok"},
        {"failure_reason": "truncated", "format_reason": "missing_json_fence"},
    ]

    out = breakdown_rows(rows)

    assert {r["kind"] for r in out} == {"failure_reason", "format_reason"}
    assert all(r["share"] == 0.5 for r in out)
    assert sum(r["count"] for r in out if r["kind"] == "failure_reason") == 2


def test_breakdown_rows_skips_absent_columns():
    # Solver logs carry no format_reason — the table must not invent one.
    out = breakdown_rows([{"failure_reason": "ok"}])

    assert {r["kind"] for r in out} == {"failure_reason"}


def test_build_run_health_has_one_row_per_iteration_and_role(paths):
    _write_log(paths, "challenger", [_challenger_row(0, "ok"), _challenger_row(1, "truncated")], iteration=1)
    _write_log(paths, "solver", [{"failure_reason": "ok", "rank_reward": 1.0}], iteration=1)
    _write_log(paths, "challenger", [_challenger_row(0, "ok")], iteration=2)

    health = build_run_health(paths)

    assert [(r["iteration"], r["role"]) for r in health] == [
        (1, "challenger"), (1, "solver"), (2, "challenger"),
    ]
    assert health[0]["ok_rate"] == 0.5
    assert health[0]["top_failure_reason"] == "truncated"
    # format_reason is challenger-only, so the solver row must omit it entirely.
    assert "top_format_reason" not in health[1]


def test_build_run_health_is_empty_without_logs(paths):
    assert build_run_health(paths) == []


# =============================================================================
# Entry points degrade quietly without W&B
# =============================================================================

def test_report_phase_noops_when_wandb_disabled(paths, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _write_log(paths, "challenger", [_challenger_row(0)])

    report_phase(paths, "challenger", max_steps=2)  # must not raise

    assert "skipping run report" in capsys.readouterr().out


def test_report_generation_noops_when_wandb_unauthenticated(paths, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "online")

    report_generation(paths, paths.prompts_json())  # missing file, no W&B — still silent


def test_report_phase_swallows_unexpected_errors(paths, monkeypatch, capsys):
    # Telemetry must never fail a training run that otherwise succeeded.
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(
        "creative_rzero.steps.report._open_summary_run",
        lambda p: (_ for _ in ()).throw(RuntimeError("wandb exploded")),
    )

    report_phase(paths, "challenger")

    assert "phase report failed" in capsys.readouterr().out
