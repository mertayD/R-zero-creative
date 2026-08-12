"""Tests for creative_rzero/orchestrator.py — the co-evolution loop, resume
state, checkpoint verification, and the prompt-generation subprocess wrapper.
All GPU/Modal/vLLM dependencies are faked; nothing here touches a network."""

import json
import subprocess
from pathlib import Path

import pytest

from creative_rzero import orchestrator
from creative_rzero.config import load
from creative_rzero.orchestrator import (
    CheckpointVerificationError,
    CoevolveState,
    PromptShortfallError,
    SftCriticJudgeError,
    aux_gpu_id,
    generate_solver_prompts,
    n_training_gpus,
    run_coevolve,
    training_gpu_ids,
    verify_checkpoint,
    wait_for_sft_critic_judge,
)
from creative_rzero.paths import RunPaths

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_EXP = REPO_ROOT / "configs" / "exp" / "validation_tiny.yaml"


@pytest.fixture
def cfg():
    return load(TINY_EXP)


@pytest.fixture
def paths(tmp_path):
    return RunPaths(tmp_path, "t29-validation-tiny", "20260807_000000", 1)


# ---------------------------------------------------------------------------
# GPU layout
# ---------------------------------------------------------------------------

def test_gpu_layout_follows_n_gpus_per_node(cfg):
    assert n_training_gpus(cfg) == 1
    assert training_gpu_ids(cfg) == "0"
    assert aux_gpu_id(cfg) == 1


def test_gpu_layout_multi_gpu(cfg):
    cfg.verl["trainer"]["n_gpus_per_node"] = 4
    assert training_gpu_ids(cfg) == "0,1,2,3"
    assert aux_gpu_id(cfg) == 4


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------

def test_state_round_trip(tmp_path):
    state = CoevolveState(
        run_ts="20260807_000000",
        wandb_group="coevolve_x_20260807_000000",
        next_iter=2,
        next_phase="solver",
        current_challenger="/ckpt/challenger",
        current_solver="/ckpt/solver",
    )
    f = tmp_path / "state" / "x.json"
    state.save(f)
    assert CoevolveState.load(f) == state


def test_state_load_missing_returns_none(tmp_path):
    assert CoevolveState.load(tmp_path / "nope.json") is None


# ---------------------------------------------------------------------------
# verify_checkpoint
# ---------------------------------------------------------------------------

def _make_ckpt(dir_: Path, weights: bool = True):
    dir_.mkdir(parents=True)
    (dir_ / "config.json").write_text("{}")
    if weights:
        (dir_ / "model-00001.safetensors").write_text("fake")


def test_verify_checkpoint_ok(tmp_path):
    ckpt = tmp_path / "ckpt"
    _make_ckpt(ckpt)
    calls = []
    verify_checkpoint("test", str(ckpt), reload_fn=lambda: calls.append(1))
    assert calls == [1]  # volume reload happens before the existence check


def test_verify_checkpoint_missing_config(tmp_path):
    with pytest.raises(CheckpointVerificationError, match="config.json missing"):
        verify_checkpoint("test", str(tmp_path / "nope"))


def test_verify_checkpoint_missing_weights(tmp_path):
    ckpt = tmp_path / "ckpt"
    _make_ckpt(ckpt, weights=False)
    with pytest.raises(CheckpointVerificationError, match="no weight files"):
        verify_checkpoint("test", str(ckpt))


# ---------------------------------------------------------------------------
# generate_solver_prompts (subprocess wrapper)
# ---------------------------------------------------------------------------

def _fake_prompts_json(n):
    return json.dumps({"prompts": [{"prompt_id": f"p{i}"} for i in range(n)], "generation_log": {}})


def test_generate_solver_prompts_happy_path(cfg, paths):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        out = paths.prompts_json(suffix=cfg.run.profile)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_fake_prompts_json(3))
        return subprocess.CompletedProcess(cmd, 0)

    out = generate_solver_prompts(cfg, paths, "/ckpt/challenger", run=fake_run)
    assert out == paths.prompts_json(suffix="smoke")
    # generation runs on the aux GPU with the CLI's STORAGE_PATH contract
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == str(aux_gpu_id(cfg))
    assert captured["env"]["STORAGE_PATH"] == str(paths.storage)
    assert "--num_samples" in captured["cmd"]
    n_idx = captured["cmd"].index("--num_samples") + 1
    assert captured["cmd"][n_idx] == str(cfg.solver.num_train + cfg.solver.num_val)


def test_generate_solver_prompts_shortfall(cfg, paths):
    def fake_run(cmd, **kwargs):
        out = paths.prompts_json(suffix=cfg.run.profile)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_fake_prompts_json(1))  # needs 3
        return subprocess.CompletedProcess(cmd, 0)

    with pytest.raises(PromptShortfallError, match="expected >= 3 prompts, got 1"):
        generate_solver_prompts(cfg, paths, "/ckpt/challenger", run=fake_run)


def test_generate_solver_prompts_subprocess_failure(cfg, paths):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1)

    with pytest.raises(PromptShortfallError, match="exited 1"):
        generate_solver_prompts(cfg, paths, "/ckpt/challenger", run=fake_run)


# ---------------------------------------------------------------------------
# run_coevolve — loop, state hand-off, resume
# ---------------------------------------------------------------------------

def _mk_phase_fns(tmp_path, log):
    """Fake phase fns that record calls and return distinct existing ckpts."""

    def mk_ckpt(name):
        d = tmp_path / "models" / name
        if not d.exists():
            _make_ckpt(d)
        return str(d)

    def challenger_fn(cfg, run_ts, iteration):
        log.append(("challenger", iteration, cfg.challenger.model_path, cfg.solver.model_path))
        return mk_ckpt(f"challenger_iter{iteration}")

    def solver_fn(cfg, run_ts, iteration, challenger_ckpt):
        log.append(("solver", iteration, cfg.solver.model_path, challenger_ckpt))
        return mk_ckpt(f"solver_iter{iteration}")

    return challenger_fn, solver_fn


def test_run_coevolve_two_iterations_chains_checkpoints(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg.run.num_iters = 2
    log = []
    challenger_fn, solver_fn = _mk_phase_fns(tmp_path, log)
    commits = []

    final_c, final_s = run_coevolve(
        cfg, tmp_path,
        train_challenger_fn=challenger_fn,
        train_solver_fn=solver_fn,
        commit_fn=lambda: commits.append(1),
    )

    assert [(r, i) for r, i, *_ in log] == [
        ("challenger", 1), ("solver", 1), ("challenger", 2), ("solver", 2),
    ]
    # iteration 1 starts from the base models
    assert log[0][2] == "Qwen/Qwen3-0.6B" and log[0][3] == "Qwen/Qwen3-0.6B"
    # solver phase 1 gets the freshly trained challenger ckpt
    assert log[1][3] == str(tmp_path / "models" / "challenger_iter1")
    # iteration 2 starts from iteration 1's merged checkpoints
    assert log[2][2] == str(tmp_path / "models" / "challenger_iter1")
    assert log[2][3] == str(tmp_path / "models" / "solver_iter1")
    assert final_c == str(tmp_path / "models" / "challenger_iter2")
    assert final_s == str(tmp_path / "models" / "solver_iter2")
    # state committed after each of the 4 phases + once on clean completion
    assert len(commits) == 5
    # clean completion removes the state file
    state_file = RunPaths(tmp_path, cfg.run.abbr, "_", 1).state_file()
    assert not state_file.exists()


def test_run_coevolve_resumes_at_solver_phase(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    log = []
    challenger_fn, solver_fn = _mk_phase_fns(tmp_path, log)

    # Pre-existing state: Phase A of iter 1 done, was about to run Phase B.
    challenger_ckpt = tmp_path / "models" / "challenger_iter1"
    _make_ckpt(challenger_ckpt)
    state_file = RunPaths(tmp_path, cfg.run.abbr, "_", 1).state_file()
    CoevolveState(
        run_ts="20260807_000000",
        wandb_group="g",
        next_iter=1,
        next_phase="solver",
        current_challenger=str(challenger_ckpt),
        current_solver="Qwen/Qwen3-0.6B",
    ).save(state_file)

    run_coevolve(
        cfg, tmp_path,
        train_challenger_fn=challenger_fn,
        train_solver_fn=solver_fn,
    )

    # Phase A skipped; solver trains against the recorded challenger ckpt.
    assert [(r, i) for r, i, *_ in log] == [("solver", 1)]
    assert log[0][3] == str(challenger_ckpt)


def test_run_coevolve_failed_phase_leaves_resumable_state(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    log = []
    challenger_fn, _ = _mk_phase_fns(tmp_path, log)

    def failing_solver(cfg, run_ts, iteration, challenger_ckpt):
        raise RuntimeError("solver phase crashed")

    with pytest.raises(RuntimeError, match="solver phase crashed"):
        run_coevolve(
            cfg, tmp_path,
            train_challenger_fn=challenger_fn,
            train_solver_fn=failing_solver,
        )

    # State survives pointing at the solver phase of iter 1, with Phase A's ckpt.
    state = CoevolveState.load(RunPaths(tmp_path, cfg.run.abbr, "_", 1).state_file())
    assert state is not None
    assert (state.next_iter, state.next_phase) == (1, "solver")
    assert state.current_challenger == str(tmp_path / "models" / "challenger_iter1")


def test_run_coevolve_verifies_checkpoints(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def bad_challenger(cfg, run_ts, iteration):
        return str(tmp_path / "does_not_exist")

    with pytest.raises(CheckpointVerificationError):
        run_coevolve(
            cfg, tmp_path,
            train_challenger_fn=bad_challenger,
            train_solver_fn=lambda *a, **k: pytest.fail("solver must not run"),
        )


# ---------------------------------------------------------------------------
# num_iters config validation
# ---------------------------------------------------------------------------

def test_num_iters_validated():
    from creative_rzero.config import ConfigError

    with pytest.raises(ConfigError, match="num_iters"):
        load(TINY_EXP, cli_args=["run.num_iters=0"])


# ---------------------------------------------------------------------------
# sft-critic judge warm-up (T3.11) — REFACTOR_PLAN.md §6.2
# ---------------------------------------------------------------------------

class _FakeHealthResponse:
    def __init__(self, ok):
        self.ok = ok


def test_wait_for_sft_critic_judge_succeeds_immediately_when_healthy(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _FakeHealthResponse(ok=True)

    monkeypatch.setattr("requests.get", fake_get)

    wait_for_sft_critic_judge("https://fake-critic.modal.run/", health_timeout_s=5, poll_interval_s=0)
    assert calls == ["https://fake-critic.modal.run/health"]


def test_wait_for_sft_critic_judge_polls_through_cold_start(monkeypatch):
    import requests

    responses = [requests.RequestException("cold"), _FakeHealthResponse(ok=False), _FakeHealthResponse(ok=True)]

    def fake_get(url, timeout):
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("creative_rzero.orchestrator.time.sleep", lambda _: None)

    wait_for_sft_critic_judge("https://fake-critic.modal.run", health_timeout_s=5, poll_interval_s=0)
    assert responses == []


def test_wait_for_sft_critic_judge_times_out(monkeypatch):
    monkeypatch.setattr("requests.get", lambda url, timeout: _FakeHealthResponse(ok=False))
    monkeypatch.setattr("creative_rzero.orchestrator.time.sleep", lambda _: None)

    with pytest.raises(SftCriticJudgeError, match="not ready after"):
        wait_for_sft_critic_judge("https://fake-critic.modal.run", health_timeout_s=0, poll_interval_s=0)


def test_run_coevolve_warms_up_sft_critic_judge_once(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg.judge.type = "sft-critic"
    cfg.judge.critic_url = "https://fake-critic.modal.run"
    cfg.run.num_iters = 2
    log = []
    challenger_fn, solver_fn = _mk_phase_fns(tmp_path, log)

    warmups = []
    monkeypatch.setattr(orchestrator, "wait_for_sft_critic_judge", lambda url: warmups.append(url))

    run_coevolve(cfg, tmp_path, train_challenger_fn=challenger_fn, train_solver_fn=solver_fn)

    # once per run_coevolve call, not once per phase/iteration.
    assert warmups == ["https://fake-critic.modal.run"]


def test_run_coevolve_skips_warmup_for_non_sft_critic_judge(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert cfg.judge.type != "sft-critic"
    log = []
    challenger_fn, solver_fn = _mk_phase_fns(tmp_path, log)

    def fail_if_called(url):
        pytest.fail("wait_for_sft_critic_judge must not be called for judge.type != sft-critic")

    monkeypatch.setattr(orchestrator, "wait_for_sft_critic_judge", fail_if_called)

    run_coevolve(cfg, tmp_path, train_challenger_fn=challenger_fn, train_solver_fn=solver_fn)
