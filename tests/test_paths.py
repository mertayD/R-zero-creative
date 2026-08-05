# Layout-pinning tests for RunPaths: each test asserts an exact string,
# transcribed from the current f-string constructions in modal_run.py, the
# two creative_{challenger,solver}_smoke.sh scripts, and the reward
# callers. The point isn't to test RunPaths' logic in the abstract — it's
# to catch any future edit (here or at a call site) that silently drifts
# the layout away from what those still-live consumers expect.
from pathlib import Path

import pytest

from creative_rzero.paths import RunPaths

STORAGE = "/storage"
ABBR = "qwen3-4b-coevolve"
RUN_TS = "20260805_120000"


@pytest.fixture
def paths() -> RunPaths:
    # Iteration 1 by default; iteration-sensitivity itself is covered
    # separately by test_iteration_changes_all_derived_paths, which builds
    # its own instances rather than mutating this fixture's.
    return RunPaths(STORAGE, ABBR, RUN_TS, iteration=1)


def test_iter_abbr(paths: RunPaths):
    assert paths.iter_abbr == "qwen3-4b-coevolve_20260805_120000_iter1"


def test_save_name(paths: RunPaths):
    assert paths.save_name("challenger") == "qwen3-4b-coevolve_20260805_120000_iter1_challenger_v1"
    assert paths.save_name("solver") == "qwen3-4b-coevolve_20260805_120000_iter1_solver_v1"


def test_save_name_rejects_unknown_role(paths: RunPaths):
    # _check_role fires from every role-taking method; save_name is just the
    # one every other method routes through (directly or via checkpoint_root).
    with pytest.raises(ValueError, match="role='judge'"):
        paths.save_name("judge")


def test_checkpoint_layout_matches_model_merger_convention(paths: RunPaths):
    assert paths.checkpoint_root("challenger") == Path(
        "/storage/models/qwen3-4b-coevolve_20260805_120000_iter1_challenger_v1"
    )
    assert paths.checkpoint_step_dir("challenger", 3) == Path(
        "/storage/models/qwen3-4b-coevolve_20260805_120000_iter1_challenger_v1/global_step_3/actor"
    )
    # model_merger.py writes its merged output to `<local_dir>/huggingface`
    assert paths.challenger_ckpt(3) == Path(
        "/storage/models/qwen3-4b-coevolve_20260805_120000_iter1_challenger_v1/global_step_3/actor/huggingface"
    )
    assert paths.solver_ckpt(1) == Path(
        "/storage/models/qwen3-4b-coevolve_20260805_120000_iter1_solver_v1/global_step_1/actor/huggingface"
    )


def test_last_ckpt_file_keyed_on_iter_abbr(paths: RunPaths):
    # Regression pin for T1.3: the bash scripts write this file keyed on the
    # abbr they were actually invoked with (iter_abbr under the orchestrator),
    # not the top-level run abbr — modal_run.py reads it back by that name.
    assert paths.last_ckpt_file("solver") == Path(
        "/storage/models/qwen3-4b-coevolve_20260805_120000_iter1_solver_last_ckpt.txt"
    )


def test_prompts_json(paths: RunPaths):
    assert paths.prompts_json() == Path(
        "/storage/generated_questions_one_shot/qwen3-4b-coevolve_20260805_120000_iter1_solver_prompts_smoke.json"
    )
    assert paths.prompts_json(suffix="full") == Path(
        "/storage/generated_questions_one_shot/qwen3-4b-coevolve_20260805_120000_iter1_solver_prompts_full.json"
    )


def test_parquet_paths(paths: RunPaths):
    # Challenger parquet has no "solver_" infix, solver parquet does — see
    # the comment on RunPaths._parquet for why this asymmetry is intentional.
    assert paths.train_parquet("challenger") == Path(
        "/storage/creative_smoke/qwen3-4b-coevolve_20260805_120000_iter1_train.parquet"
    )
    assert paths.val_parquet("challenger") == Path(
        "/storage/creative_smoke/qwen3-4b-coevolve_20260805_120000_iter1_val.parquet"
    )
    assert paths.train_parquet("solver") == Path(
        "/storage/creative_smoke/qwen3-4b-coevolve_20260805_120000_iter1_solver_train.parquet"
    )
    assert paths.val_parquet("solver") == Path(
        "/storage/creative_smoke/qwen3-4b-coevolve_20260805_120000_iter1_solver_val.parquet"
    )


def test_rollout_log(paths: RunPaths):
    assert paths.rollout_log("challenger") == Path(
        "/storage/reward_logs/qwen3-4b-coevolve_20260805_120000_iter1_challenger_v1.jsonl"
    )


def test_judge_cache(paths: RunPaths):
    assert paths.judge_cache() == Path("/storage/judge_cache")


def test_state_file_keyed_on_run_abbr_not_iter_abbr(paths: RunPaths):
    # One state file spans every iteration of a run — modal_run.py's
    # train_creative_coevolve keys it on the run-level `abbr` param, before
    # any `_iter{n}` suffix is added.
    assert paths.state_file() == Path("/storage/coevolve_state/qwen3-4b-coevolve.json")


def test_manifest(paths: RunPaths):
    assert paths.run_dir() == Path("/storage/runs/qwen3-4b-coevolve_20260805_120000")
    assert paths.manifest() == Path("/storage/runs/qwen3-4b-coevolve_20260805_120000/manifest.json")


def test_iteration_changes_all_derived_paths():
    p1 = RunPaths(STORAGE, ABBR, RUN_TS, iteration=1)
    p2 = RunPaths(STORAGE, ABBR, RUN_TS, iteration=2)

    assert p1.iter_abbr != p2.iter_abbr
    assert p1.checkpoint_root("solver") != p2.checkpoint_root("solver")
    # run-spanning paths must NOT change between iterations of the same run
    assert p1.state_file() == p2.state_file()
    assert p1.manifest() == p2.manifest()
