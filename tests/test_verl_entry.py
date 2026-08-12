"""Tests for creative_rzero/rewards/verl_entry.py — the single reward
entrypoint verl's `worker.reward.reward_function` points at.

The legacy caller modules import `vllm` transitively (via
evaluation/writing_bench/evaluator); like tests/test_generate_prompts.py, a
fake `vllm` module is injected so the dispatch path is exercised without a
GPU. The happy-path test performs the REAL caller-module import — that's
the regression this file exists for: base.yaml pointed at a file that
didn't exist, and nothing failed until a paid GPU run."""

import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_EXP = REPO_ROOT / "configs" / "exp" / "validation_tiny.yaml"


@pytest.fixture
def fake_vllm_module(monkeypatch):
    fake = types.ModuleType("vllm")
    fake.LLM = object
    fake.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", fake)
    return fake


@pytest.fixture
def verl_entry(monkeypatch):
    """Import verl_entry the way verl does (fresh module, standalone), with a
    clean impl cache per test."""
    import creative_rzero.rewards.verl_entry as ve

    importlib.reload(ve)
    return ve


@pytest.fixture
def resolved_config_env(monkeypatch, tmp_path):
    """A resolved config on disk + EXPERIMENT_CONFIG_PATH pointing at it."""
    from creative_rzero.config import load, save_resolved

    cfg = load(TINY_EXP)
    out = save_resolved(cfg, tmp_path)
    monkeypatch.setenv("EXPERIMENT_CONFIG_PATH", str(out))
    return cfg


def test_compute_score_requires_role(verl_entry):
    with pytest.raises(ValueError, match="without a role"):
        verl_entry.compute_score(["x"], ["y"])


def test_invalid_role_rejected(verl_entry):
    with pytest.raises(ValueError, match="must be one of"):
        verl_entry.compute_score(["x"], ["y"], role="judge")


def test_missing_config_path_is_a_pointed_error(verl_entry, monkeypatch):
    monkeypatch.delenv("EXPERIMENT_CONFIG_PATH", raising=False)
    with pytest.raises(RuntimeError, match="EXPERIMENT_CONFIG_PATH is not set"):
        verl_entry.compute_score(["x"], ["y"], role="solver")


def test_unimplemented_strategy_rejected(verl_entry, resolved_config_env, monkeypatch, tmp_path):
    """A config asking for a strategy the legacy callers don't implement must
    fail loudly, not silently score with the wrong formula."""
    from creative_rzero.config import load, save_resolved

    cfg = load(TINY_EXP)
    cfg.rewards.solver.type = "zscore"  # bypass load-time validation deliberately
    out = save_resolved(cfg, tmp_path / "alt")
    monkeypatch.setenv("EXPERIMENT_CONFIG_PATH", str(out))
    with pytest.raises(ValueError, match="zscore.*not available"):
        verl_entry.compute_score(["x"], ["y"], role="solver")


def test_solver_dispatch_reaches_real_caller(verl_entry, resolved_config_env, fake_vllm_module, monkeypatch):
    """_get_impl('solver') must import the real creative_solver_caller module
    and bridge config values into the env names it reads at import time."""
    monkeypatch.setenv("REMOTE_REPO_PATH", str(REPO_ROOT))
    # Force a fresh import so import-time env reads see the bridged values.
    for name in list(sys.modules):
        if "creative_solver_caller" in name or "creative_writing_caller" in name:
            monkeypatch.delitem(sys.modules, name)

    impl = verl_entry._get_impl("solver")
    assert callable(impl)
    assert impl.__module__ == "examples.reward_function.creative_solver_caller"

    import os

    cfg = resolved_config_env
    assert os.environ["WB_JUDGE_TYPE"] == cfg.judge.type
    assert os.environ["WB_JUDGE_MODEL"] == cfg.judge.model
    assert os.environ["SOLVER_MAX_RESPONSE_LENGTH"] == str(cfg.solver.max_response_length)
    assert os.environ["CREATIVE_SOLVER_PORT"] == str(cfg.challenger.solver_query.port)
    assert os.environ["CREATIVE_LOW_QUALITY_THRESHOLD"] == str(cfg.judge.low_quality_threshold)
    assert os.environ["CREATIVE_SCORER_MAX_WORKERS"] == str(cfg.judge.max_workers)

    # cached on second call — no re-import
    assert verl_entry._get_impl("solver") is impl


def test_challenger_dispatch_reaches_real_caller(verl_entry, resolved_config_env, fake_vllm_module, monkeypatch):
    monkeypatch.setenv("REMOTE_REPO_PATH", str(REPO_ROOT))
    impl = verl_entry._get_impl("challenger")
    assert impl.__module__ == "examples.reward_function.creative_writing_caller"

    import os

    cfg = resolved_config_env
    assert os.environ["CHALLENGER_MAX_RESPONSE_LENGTH"] == str(cfg.challenger.max_response_length)
    assert os.environ["CREATIVE_SOLVER_MAX_TOKENS"] == str(cfg.challenger.solver_query.max_tokens)


def test_mock_judge_type_routes_caller_to_mock_agent(verl_entry, fake_vllm_module, monkeypatch, tmp_path):
    """judge.type=mock must make the reward caller build a MockJudgeAgent
    instead of ClaudeAgent — the entire point being a training/reward run
    that needs no PERPLEXITY_API_KEY at all."""
    from creative_rzero.config import load, save_resolved

    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("REMOTE_REPO_PATH", str(REPO_ROOT))
    for name in list(sys.modules):
        if "creative_solver_caller" in name:
            monkeypatch.delitem(sys.modules, name)

    cfg = load(TINY_EXP, cli_args=["judge.type=mock"])
    out = save_resolved(cfg, tmp_path)
    monkeypatch.setenv("EXPERIMENT_CONFIG_PATH", str(out))

    impl_module = verl_entry._get_impl("solver").__module__
    caller = sys.modules[impl_module]
    agent = caller._get_agent()

    from batch_eval_agent import BatchEvalAgent
    from evaluator import MockJudgeAgent

    assert isinstance(agent, BatchEvalAgent)
    assert isinstance(agent.agent, MockJudgeAgent)


def test_sft_critic_judge_type_routes_caller_to_critic_agent(verl_entry, fake_vllm_module, monkeypatch, tmp_path):
    """judge.type=sft-critic must make the reward caller build a
    PerCriterionEvalAgent(CriticServerAgent), not BatchEvalAgent(ClaudeAgent)
    — and _bridge_config_to_env must carry WB_CRITIC_URL/WB_CRITIC_MODEL."""
    from creative_rzero.config import load, save_resolved

    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("REMOTE_REPO_PATH", str(REPO_ROOT))
    for name in list(sys.modules):
        if "creative_solver_caller" in name:
            monkeypatch.delitem(sys.modules, name)

    cfg = load(
        TINY_EXP,
        cli_args=["judge.type=sft-critic", "judge.critic_url=https://fake-critic.modal.run"],
    )
    out = save_resolved(cfg, tmp_path)
    monkeypatch.setenv("EXPERIMENT_CONFIG_PATH", str(out))

    impl_module = verl_entry._get_impl("solver").__module__
    caller = sys.modules[impl_module]
    agent = caller._get_agent()

    from batch_eval_agent import PerCriterionEvalAgent
    from evaluator import CriticServerAgent

    assert isinstance(agent, PerCriterionEvalAgent)
    assert isinstance(agent.agent, CriticServerAgent)

    import os

    assert os.environ["WB_CRITIC_URL"] == "https://fake-critic.modal.run"
    assert os.environ["WB_CRITIC_MODEL"] == cfg.judge.critic_model


@pytest.mark.parametrize("bogus_judge_type", [None, "", "Claude", "gpt-4", "sft_critic"])
def test_unrecognized_judge_type_raises_no_silent_claude_fallback(
    verl_entry, resolved_config_env, fake_vllm_module, monkeypatch, bogus_judge_type
):
    """An unset/typo'd/unrecognized WB_JUDGE_TYPE must fail loud in the
    reward caller itself, not silently fall through to ClaudeAgent — config
    validation (creative_rzero/config.py VALID_JUDGE_TYPES) is
    defense-in-depth, not the only guard against a wrong/expensive judge
    getting used unnoticed. Imports both caller modules the normal way
    (through a valid config, via verl_entry._get_impl) so this exercises
    the real modules already in sys.modules, then overrides WB_JUDGE_TYPE
    to the bogus value afterward — mimicking a stray manual override rather
    than anything the config bridge itself would ever produce."""
    monkeypatch.setenv("REMOTE_REPO_PATH", str(REPO_ROOT))
    for name in list(sys.modules):
        if "creative_solver_caller" in name or "creative_writing_caller" in name:
            monkeypatch.delitem(sys.modules, name)

    solver_module = sys.modules[verl_entry._get_impl("solver").__module__]
    challenger_module = sys.modules[verl_entry._get_impl("challenger").__module__]

    if bogus_judge_type is None:
        monkeypatch.delenv("WB_JUDGE_TYPE", raising=False)
    else:
        monkeypatch.setenv("WB_JUDGE_TYPE", bogus_judge_type)

    for caller in (solver_module, challenger_module):
        caller._agent = None  # reset the lazy singleton between parametrize cases
        with pytest.raises(ValueError, match="not a recognized judge type"):
            caller._get_agent()


def test_materialized_verl_config_carries_role_and_abs_reward_path(tmp_path):
    """steps/train_verl.py's side of the contract: reward_function_kwargs.role
    is set and reward_function is an existing absolute path."""
    from omegaconf import OmegaConf

    from creative_rzero.config import load
    from creative_rzero.paths import RunPaths
    from creative_rzero.steps.train_verl import materialize_verl_config

    cfg = load(TINY_EXP)
    paths = RunPaths(tmp_path, "abbr", "20260807_000000", 1)
    out = materialize_verl_config(cfg, paths, "solver", tmp_path / "run")
    verl_cfg = OmegaConf.load(out)

    assert verl_cfg.worker.reward.reward_function_kwargs == {"role": "solver"}
    fn_path, _, fn_name = verl_cfg.worker.reward.reward_function.rpartition(":")
    assert fn_name == "compute_score"
    assert Path(fn_path).is_absolute()
    assert Path(fn_path).exists()
    assert fn_path.endswith("creative_rzero/rewards/verl_entry.py")


def test_materialize_rejects_missing_reward_file(tmp_path):
    from creative_rzero.config import load
    from creative_rzero.paths import RunPaths
    from creative_rzero.steps.train_verl import materialize_verl_config

    cfg = load(TINY_EXP)
    cfg.verl["worker"]["reward"]["reward_function"] = "./does/not/exist.py:compute_score"
    paths = RunPaths(tmp_path, "abbr", "20260807_000000", 1)
    with pytest.raises(FileNotFoundError, match="does/not/exist.py"):
        materialize_verl_config(cfg, paths, "solver", tmp_path / "run")
