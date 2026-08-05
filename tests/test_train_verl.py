from pathlib import Path

import pytest
from omegaconf import OmegaConf

from creative_rzero.config import load
from creative_rzero.paths import RunPaths
from creative_rzero.steps.train_verl import (
    TrainingFailedError,
    launch_training,
    materialize_verl_config,
)

EXAMPLE_EXP = "configs/exp/example.yaml"


@pytest.fixture
def cfg():
    return load(
        EXAMPLE_EXP,
        cli_args=[
            "solver.max_steps=6",
            "solver.rollout_n=8",
            "solver.num_train=8",
            "solver.num_val=2",
            "solver.max_response_length=4096",
        ],
    )


@pytest.fixture
def paths(tmp_path) -> RunPaths:
    return RunPaths(tmp_path / "storage", "example", "20260805_120000", iteration=1)


def test_materialize_verl_config_fills_role_placeholders(cfg, paths, tmp_path):
    out_path = materialize_verl_config(cfg, paths, "solver", tmp_path / "run_dir")

    assert out_path == tmp_path / "run_dir" / "verl_config_solver.yaml"
    resolved = OmegaConf.load(out_path)

    assert resolved.data.train_files == str(paths.train_parquet("solver"))
    assert resolved.data.val_files == str(paths.val_parquet("solver"))
    assert resolved.data.max_response_length == 4096
    assert resolved.data.rollout_batch_size == 8
    assert resolved.data.val_batch_size == 2
    assert resolved.data.seed == cfg.run.seed
    assert resolved.worker.actor.model.model_path == cfg.solver.model_path
    assert resolved.worker.actor.global_batch_size == 8
    assert resolved.worker.rollout.n == 8
    assert resolved.trainer.experiment_name == paths.save_name("solver")
    assert resolved.trainer.save_checkpoint_path == str(paths.checkpoint_root("solver"))
    assert resolved.trainer.max_steps == 6
    # passthrough keys are untouched
    assert resolved.data.format_prompt == "./examples/format_prompt/creative.jinja"


def test_materialize_verl_config_rejects_unknown_role(cfg, paths, tmp_path):
    with pytest.raises(ValueError, match="role='judge'"):
        materialize_verl_config(cfg, paths, "judge", tmp_path / "run_dir")


class _FakeProc:
    def __init__(self, lines, returncode):
        self.stdout = iter(lines)
        self._returncode = returncode

    def wait(self):
        return self._returncode


def test_launch_training_returns_config_path_on_success(cfg, paths, tmp_path):
    calls = {}

    def fake_popen(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["env"] = kwargs["env"]
        return _FakeProc(["step 1\n", "step 2\n"], 0)

    run_dir = tmp_path / "run_dir"
    out_path = launch_training(cfg, paths, "solver", run_dir, popen=fake_popen)

    assert out_path == run_dir / "verl_config_solver.yaml"
    assert calls["cmd"][0:3] == [calls["cmd"][0], "-m", "verl.trainer.main"]
    assert calls["cmd"][3] == f"config={out_path}"
    env = calls["env"]
    assert env["STORAGE_PATH"] == str(paths.storage)
    assert env["EXPERIMENT_CONFIG_PATH"] == str(run_dir / "resolved_config.yaml")
    assert env["VERL_EXPERIMENT_NAME"] == paths.save_name("solver")
    assert env["COEVOLVE_ITERATION"] == "1"
    assert (run_dir / "resolved_config.yaml").exists()


def test_launch_training_raises_with_tail_on_nonzero_exit(cfg, paths, tmp_path):
    def fake_popen(cmd, **kwargs):
        return _FakeProc(["all good\n", "boom\n"], 1)

    with pytest.raises(TrainingFailedError, match="exited 1 for role=solver"):
        launch_training(cfg, paths, "solver", tmp_path / "run_dir", popen=fake_popen)
