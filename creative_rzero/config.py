"""Typed load/merge/validate/dump for creative-writing experiment configs.

An experiment config is `configs/base.yaml` overridden by a small
`configs/exp/<name>.yaml` file, optionally overridden again by CLI dotlist
args — the same merge order verl.trainer.main itself uses for its own
config (structured defaults -> file -> CLI), so override semantics agree
on both sides of the `steps/train_verl.py` boundary.

The `verl` block is intentionally left as an untyped pass-through dict
rather than verl's own `PPOConfig` dataclass: importing verl pulls in
torch/transformers/vllm at module load time (verl.workers.reward and
verl.workers.rollout import them eagerly), which would defeat the point of
validating a config in seconds on a laptop. `steps/train_verl.py` — which
already needs verl importable to launch training — is where the verl
sub-block gets its own schema validation, immediately before it's
materialized into the run dir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

VALID_PROFILES = ("dryrun", "smoke", "small", "full")
# TODO(dayanc): To add small SFTed judge support.
VALID_JUDGE_TYPES = ("claude", "mock")
# Stand-in for the Phase-3 reward registry (T3.3), which doesn't exist yet.
# Only the strategies that actually exist today are accepted. Once
# rewards/registry.py lands, swap these for `registry.SOLVER_REWARDS.keys()`
# / `registry.CHALLENGER_REWARDS.keys()`.
# TODO(T3.4): add "zscore", "raw", "pairwise" once those solver strategies exist.
VALID_SOLVER_REWARDS = ("rank",)
# TODO(T3.5): add "variance", "target_band" once those challenger strategies exist.
VALID_CHALLENGER_REWARDS = ("uncertainty",)


class ConfigError(ValueError):
    """Raised when an experiment config fails validation at load time."""


@dataclass
class RunConfig:
    abbr: Optional[str] = None  # required, set per run
    profile: str = "smoke"  # dryrun | smoke | small | full
    seed: int = 42
    num_iters: int = 1  # co-evolution rounds (challenger phase + solver phase each)


@dataclass
class SolverQueryConfig:
    port: int = 5000
    max_tokens: int = 32768
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0


@dataclass
class ChallengerConfig:
    model_path: Optional[str] = None  # required, set per run
    max_steps: int = 4
    rollout_n: int = 4
    num_train: int = 8
    num_val: int = 2
    max_response_length: int = 4096
    solver_query: SolverQueryConfig = field(default_factory=SolverQueryConfig)


@dataclass
class SolverConfig:
    model_path: Optional[str] = None  # required, set per run
    max_steps: int = 4
    rollout_n: int = 4
    num_train: int = 8
    num_val: int = 2
    max_response_length: int = 4096


@dataclass
class JudgeConfig:
    type: str = "claude"  # claude | mock
    model: str = "claude-sonnet-5"
    max_tokens: int = 8192
    http_max_retry_attempts: int = 5
    max_workers: int = 4
    truncation_margin_tokens: int = 16
    low_quality_threshold: float = 2.0


@dataclass
class RewardSelector:
    type: str = "rank"


@dataclass
class RewardsConfig:
    solver: RewardSelector = field(default_factory=lambda: RewardSelector(type="rank"))
    challenger: RewardSelector = field(default_factory=lambda: RewardSelector(type="uncertainty"))


@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    challenger: ChallengerConfig = field(default_factory=ChallengerConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    verl: dict[str, Any] = field(default_factory=dict)


def _validate(config: ExperimentConfig) -> None:
    errors: list[str] = []

    if not config.run.abbr:
        errors.append("run.abbr is required (short name used to build checkpoint/run directory names)")
    if config.run.profile not in VALID_PROFILES:
        errors.append(f"run.profile={config.run.profile!r} must be one of {VALID_PROFILES}")
    if config.run.num_iters < 1:
        errors.append(f"run.num_iters must be >= 1, got {config.run.num_iters}")

    for role in ("challenger", "solver"):
        role_cfg = getattr(config, role)
        if role_cfg.max_steps < 1:
            errors.append(f"{role}.max_steps must be >= 1, got {role_cfg.max_steps}")
        if role_cfg.rollout_n < 2:
            errors.append(
                f"{role}.rollout_n must be >= 2 (a group of 1 can't be ranked or compared), got {role_cfg.rollout_n}"
            )

    if config.judge.type not in VALID_JUDGE_TYPES:
        errors.append(f"judge.type={config.judge.type!r} must be one of {VALID_JUDGE_TYPES}")
    elif config.judge.type == "claude" and not (config.judge.model or "").startswith("claude-"):
        errors.append(f"judge.model={config.judge.model!r} doesn't look like a Claude model id (expected a 'claude-' prefix)")

    if config.rewards.solver.type not in VALID_SOLVER_REWARDS:
        errors.append(f"rewards.solver.type={config.rewards.solver.type!r} must be one of {VALID_SOLVER_REWARDS}")
    if config.rewards.challenger.type not in VALID_CHALLENGER_REWARDS:
        errors.append(
            f"rewards.challenger.type={config.rewards.challenger.type!r} must be one of {VALID_CHALLENGER_REWARDS}"
        )

    if errors:
        raise ConfigError("invalid experiment config:\n  - " + "\n  - ".join(errors))


def load(exp_path: Optional[str | Path] = None, cli_args: Optional[list[str]] = None) -> ExperimentConfig:
    """Load `base.yaml`, override with `exp_path`, override with `cli_args`.

    `cli_args` is a dotlist like `["challenger.max_steps=6", "run.abbr=foo"]`
    (the same syntax `OmegaConf.from_cli()` parses from `sys.argv`).

    Raises ConfigError on an unknown key anywhere outside the `verl` block,
    or a value that fails `_validate` (e.g. `rollout_n: 1`).
    """
    schema = OmegaConf.structured(ExperimentConfig())

    merged = OmegaConf.merge(schema, OmegaConf.load(BASE_CONFIG_PATH))
    if exp_path is not None:
        merged = OmegaConf.merge(merged, OmegaConf.load(exp_path))
    if cli_args:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(cli_args))

    config: ExperimentConfig = OmegaConf.to_object(merged)
    _validate(config)
    return config


def save_resolved(config: ExperimentConfig, run_dir: str | Path) -> Path:
    """Write the fully resolved config to `run_dir/resolved_config.yaml`.

    The reviewable, reproducible record of exactly what a run used.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.structured(config), out_path)
    return out_path
