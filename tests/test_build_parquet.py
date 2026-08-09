import json

import pandas as pd
import pytest

from creative_rzero.config import load
from creative_rzero.paths import RunPaths
from creative_rzero.steps.build_parquet import build_challenger_parquet, build_solver_parquet

EXAMPLE_EXP = "configs/exp/example.yaml"

# Transcribed from creative_solver_smoke.sh Step 1's heredoc (the schema this
# ports) and creative_challenger_smoke.sh Step 0's heredoc, respectively.
SOLVER_COLUMNS = [
    "problem", "answer", "prompt_id", "domain", "domain_name", "subdomain",
    "num_criteria", "criteria_json", "requirements_style", "requirements_format",
    "requirements_length", "guidance_applied_json", "num_guidance_applied",
    "format_score", "language", "seed", "iteration", "run_id", "challenger_checkpoint",
]
CHALLENGER_COLUMNS = [
    "problem", "answer", "domain", "domain_name", "subdomain", "seed",
    "iteration", "run_id", "challenger_init_model", "solver_oracle_model",
]


@pytest.fixture
def paths(tmp_path) -> RunPaths:
    return RunPaths(tmp_path, "example", "20260805_120000", iteration=1)


def _writing_prompt(i: int, *, empty_query=False, no_criteria=False) -> dict:
    return {
        "prompt_id": str(i),
        "domain": "D1",
        "domain_name": "Fiction",
        "subdomain": "short story",
        "query": "" if empty_query else f"write a story {i}",
        "criteria": [] if no_criteria else [{"name": "clarity"}],
        "requirements": {"style": "vivid", "format": None, "length": "short"},
        "guidance_applied": ["show-dont-tell"],
        "format_score": 1,
        "language": "English",
        "seed": 42,
    }


def test_build_solver_parquet_schema_and_row_count(paths, tmp_path):
    cfg = load(EXAMPLE_EXP, cli_args=["solver.num_train=3", "solver.num_val=1"])
    prompts_json = tmp_path / "prompts.json"
    prompts_json.write_text(json.dumps({"prompts": [_writing_prompt(i) for i in range(4)]}))

    train_path, val_path = build_solver_parquet(
        prompts_json, "/storage/models/challenger_v1/huggingface", cfg, paths
    )

    assert train_path == paths.train_parquet("solver")
    assert val_path == paths.val_parquet("solver")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    assert list(train_df.columns) == SOLVER_COLUMNS
    assert len(train_df) == 3
    assert len(val_df) == 1
    assert train_df.iloc[0]["challenger_checkpoint"] == "/storage/models/challenger_v1/huggingface"
    assert train_df.iloc[0]["iteration"] == 1
    assert json.loads(train_df.iloc[0]["answer"])["prompt_id"] == "0"


def test_build_solver_parquet_skips_invalid_rows(paths, tmp_path):
    cfg = load(EXAMPLE_EXP, cli_args=["solver.num_train=2", "solver.num_val=1"])
    prompts = [
        _writing_prompt(0),
        _writing_prompt(1, empty_query=True),
        _writing_prompt(2, no_criteria=True),
        _writing_prompt(3),
        _writing_prompt(4),
    ]
    prompts_json = tmp_path / "prompts.json"
    prompts_json.write_text(json.dumps({"prompts": prompts}))

    train_path, val_path = build_solver_parquet(prompts_json, "ckpt", cfg, paths)

    assert len(pd.read_parquet(train_path)) == 2
    assert len(pd.read_parquet(val_path)) == 1


def test_build_solver_parquet_raises_on_shortfall(paths, tmp_path):
    cfg = load(EXAMPLE_EXP, cli_args=["solver.num_train=3", "solver.num_val=1"])
    prompts_json = tmp_path / "prompts.json"
    prompts_json.write_text(json.dumps({"prompts": [_writing_prompt(0)]}))

    with pytest.raises(ValueError, match="need 4 valid prompts, got 1"):
        build_solver_parquet(prompts_json, "ckpt", cfg, paths)


def test_build_challenger_parquet_schema_and_row_count(paths):
    cfg = load(EXAMPLE_EXP, cli_args=["challenger.num_train=3", "challenger.num_val=2"])

    train_path, val_path = build_challenger_parquet(cfg, paths)

    assert train_path == paths.train_parquet("challenger")
    assert val_path == paths.val_parquet("challenger")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    assert list(train_df.columns) == CHALLENGER_COLUMNS
    assert len(train_df) == 3
    assert len(val_df) == 2
    assert (train_df["challenger_init_model"] == cfg.challenger.model_path).all()
    assert (train_df["solver_oracle_model"] == cfg.solver.model_path).all()
    # answer carries the logging-metadata payload (the challenger has no
    # reference answer) — it must mirror the row's own columns.
    for _, row in train_df.iterrows():
        meta = json.loads(row["answer"])
        assert meta["domain"] == row["domain"]
        assert meta["domain_name"] == row["domain_name"]
        assert meta["subdomain"] == row["subdomain"]
        assert meta["problem"] == row["problem"]


def test_build_challenger_parquet_is_deterministic_for_fixed_seed(paths):
    cfg = load(EXAMPLE_EXP, cli_args=["run.seed=7", "challenger.num_train=5", "challenger.num_val=0"])

    train_path_1, _ = build_challenger_parquet(cfg, paths)
    df1 = pd.read_parquet(train_path_1)[["domain", "subdomain"]]

    train_path_2, _ = build_challenger_parquet(cfg, paths)
    df2 = pd.read_parquet(train_path_2)[["domain", "subdomain"]]

    assert df1.equals(df2)
