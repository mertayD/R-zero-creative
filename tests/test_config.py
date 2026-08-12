import pytest
from omegaconf import OmegaConf

from creative_rzero.config import ConfigError, ExperimentConfig, load, save_resolved

EXAMPLE_EXP = "configs/exp/example.yaml"


def test_load_base_plus_example_succeeds():
    config = load(EXAMPLE_EXP)

    assert isinstance(config, ExperimentConfig)
    assert config.run.abbr == "example"
    assert config.challenger.model_path == "Qwen/Qwen3-4B"
    # values not overridden fall through from base.yaml
    assert config.challenger.max_steps == 4
    assert config.rewards.solver.type == "rank"
    # the verl block passes through untouched, still containing base.yaml's keys
    assert config.verl["data"]["format_prompt"] == "./examples/format_prompt/creative.jinja"


def test_cli_dotlist_overrides_file_and_base():
    config = load(EXAMPLE_EXP, cli_args=["challenger.max_steps=9"])

    assert config.challenger.max_steps == 9


def test_typoed_key_raises():
    with pytest.raises(Exception):
        load(cli_args=["run.abbr=x", "challenger.mode_path=typo"])


def test_rollout_n_one_raises_pointed_message():
    with pytest.raises(ConfigError, match="solver.rollout_n must be >= 2"):
        load(EXAMPLE_EXP, cli_args=["solver.rollout_n=1"])


def test_missing_abbr_raises():
    with pytest.raises(ConfigError, match="run.abbr is required"):
        load(cli_args=["challenger.model_path=x", "solver.model_path=x"])


def test_unknown_reward_type_raises():
    with pytest.raises(ConfigError, match="rewards.solver.type"):
        load(EXAMPLE_EXP, cli_args=["rewards.solver.type=nonsense"])


def test_unknown_judge_type_raises():
    with pytest.raises(ConfigError, match="judge.type"):
        load(EXAMPLE_EXP, cli_args=["judge.type=nonsense"])


def test_sft_critic_without_url_raises_at_load_time():
    """judge.type=sft-critic with no critic_url must fail at config.load()
    (startup) — before any GPU/run infra is touched, not as the first judge
    call's crash an hour into a run."""
    with pytest.raises(ConfigError, match="judge.critic_url"):
        load(EXAMPLE_EXP, cli_args=["judge.type=sft-critic"])


def test_sft_critic_with_non_http_url_raises():
    with pytest.raises(ConfigError, match="judge.critic_url"):
        load(EXAMPLE_EXP, cli_args=["judge.type=sft-critic", "judge.critic_url=not-a-url"])


def test_sft_critic_with_url_loads_successfully():
    config = load(
        EXAMPLE_EXP,
        cli_args=["judge.type=sft-critic", "judge.critic_url=https://example--critic-judge-server.modal.run"],
    )
    assert config.judge.type == "sft-critic"
    assert config.judge.critic_url == "https://example--critic-judge-server.modal.run"
    assert config.judge.critic_model == "writingbench-critic-qwen-7b"


def test_save_resolved_writes_reloadable_yaml(tmp_path):
    config = load(EXAMPLE_EXP)

    out_path = save_resolved(config, tmp_path / "run_dir")

    assert out_path == tmp_path / "run_dir" / "resolved_config.yaml"
    reloaded = OmegaConf.load(out_path)
    assert reloaded.run.abbr == "example"
    assert reloaded.verl.data.format_prompt == "./examples/format_prompt/creative.jinja"
