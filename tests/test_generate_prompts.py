import json

import pytest

from creative_rzero.config import load
from creative_rzero.paths import RunPaths
from creative_rzero.steps.generate_prompts import PromptGenerationError, generate_prompts

EXAMPLE_EXP = "configs/exp/example.yaml"


def _fake_batch(n: int) -> dict:
    return {
        "batch_id": "b0",
        "generation_log": {"total_generated": n, "total_attempted": n, "skipped": 0},
        "prompts": [{"prompt_id": str(i), "query": f"q{i}", "criteria": [{"name": "c"}]} for i in range(n)],
    }


@pytest.fixture
def paths(tmp_path) -> RunPaths:
    return RunPaths(tmp_path, "example", "20260805_120000", iteration=1)


def test_generate_prompts_writes_expected_command_and_returns_path(paths):
    cfg = load(EXAMPLE_EXP, cli_args=["solver.num_train=3", "solver.num_val=1"])

    captured_cmd = {}

    def fake_run(cmd):
        captured_cmd["cmd"] = cmd
        out_path = paths.prompts_json(suffix=cfg.run.profile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_fake_batch(4)))

    result = generate_prompts(cfg, paths, "/storage/models/challenger/huggingface", run=fake_run)

    assert result == paths.prompts_json(suffix=cfg.run.profile)
    cmd = captured_cmd["cmd"]
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "/storage/models/challenger/huggingface"
    assert cmd[cmd.index("--num_samples") + 1] == "4"
    assert cmd[cmd.index("--seed") + 1] == str(cfg.run.seed)
    assert cmd[cmd.index("--save_name") + 1] == f"{paths.iter_abbr}_solver_prompts"
    assert cmd[cmd.index("--suffix") + 1] == cfg.run.profile


def test_generate_prompts_raises_on_shortfall(paths):
    cfg = load(EXAMPLE_EXP, cli_args=["solver.num_train=3", "solver.num_val=1"])

    def fake_run(cmd):
        out_path = paths.prompts_json(suffix=cfg.run.profile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_fake_batch(2)))

    with pytest.raises(PromptGenerationError, match="expected >= 4 prompts, got 2"):
        generate_prompts(cfg, paths, "/storage/models/challenger/huggingface", run=fake_run)


def test_generate_prompts_raises_if_output_missing(paths):
    cfg = load(EXAMPLE_EXP)

    with pytest.raises(PromptGenerationError, match="did not produce"):
        generate_prompts(cfg, paths, "/storage/models/challenger/huggingface", run=lambda cmd: None)
