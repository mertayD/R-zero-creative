"""steps/train_verl.py — resolved-config VERL launcher.

Replaces the two ~25-line dotted-arg walls at the bottom of
`creative_{challenger,solver}_smoke.sh` with:

  1. `materialize_verl_config` — fills the role-specific placeholders left
     `null` in `configs/base.yaml`'s `verl` block (train/val parquet paths,
     rollout size, checkpoint dir, ...) and writes the result to
     `run_dir/verl_config_{role}.yaml`. That file is the exact config VERL
     consumes — a reviewable, diffable artifact per run, and enough on its
     own to reproduce the run (`config=<that file>`, no CLI overrides).
  2. `launch_training` — launches `python -m verl.trainer.main
     config=<that file>` as a subprocess with a small, explicit env instead
     of inheriting the caller's whole environment, streams its output, and
     raises with the last 50 lines on a nonzero exit.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Callable

from omegaconf import OmegaConf

from creative_rzero.config import ExperimentConfig, save_resolved
from creative_rzero.paths import RunPaths

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The only environment verl.trainer.main and the reward entrypoint it calls
# actually need: interpreter/model-cache plumbing, GPU selection, and
# whichever service keys the judge/W&B logging use. Everything else the old
# bash scripts inherited implicitly from the caller's shell is deliberately
# left out — the class-A "silent default" bug came from exactly that kind of
# ambient, unreviewed env contract.
_FORWARDED_ENV_VARS = (
    "PATH",
    "HOME",
    "CUDA_VISIBLE_DEVICES",
    "WANDB_MODE",
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_RUN_ID",
    "WANDB_RUN_GROUP",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "HF_HOME",
    "HF_HUB_ENABLE_HF_TRANSFER",
)
_TAIL_LINES = 50


class TrainingFailedError(RuntimeError):
    """Raised when the verl.trainer.main subprocess exits nonzero."""


def materialize_verl_config(
    cfg: ExperimentConfig,
    paths: RunPaths,
    role: str,
    run_dir: str | Path,
) -> Path:
    """Resolve `cfg.verl`'s role-specific placeholders and write the result
    to `run_dir/verl_config_{role}.yaml`.

    Every `null` in `configs/base.yaml`'s verl block that depends on which
    role is training (data paths, rollout size, checkpoint dir, max_steps)
    gets filled in from `cfg.challenger`/`cfg.solver` and `paths` here;
    everything else in `cfg.verl` passes through unchanged.
    """
    if role not in ("challenger", "solver"):
        raise ValueError(f"role={role!r} must be 'challenger' or 'solver'")

    role_cfg = getattr(cfg, role)
    verl_cfg = copy.deepcopy(cfg.verl)

    data = verl_cfg["data"]
    data["train_files"] = str(paths.train_parquet(role))
    data["val_files"] = str(paths.val_parquet(role))
    data["max_response_length"] = role_cfg.max_response_length
    data["rollout_batch_size"] = role_cfg.num_train
    data["val_batch_size"] = role_cfg.num_val
    data["seed"] = cfg.run.seed

    verl_cfg["worker"]["actor"]["model"]["model_path"] = role_cfg.model_path
    verl_cfg["worker"]["actor"]["global_batch_size"] = role_cfg.num_train
    verl_cfg["worker"]["rollout"]["n"] = role_cfg.rollout_n

    trainer = verl_cfg["trainer"]
    trainer["experiment_name"] = paths.save_name(role)
    trainer["save_checkpoint_path"] = str(paths.checkpoint_root(role))
    trainer["max_steps"] = role_cfg.max_steps

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"verl_config_{role}.yaml"
    OmegaConf.save(OmegaConf.create(verl_cfg), out_path)
    return out_path


def launch_training(
    cfg: ExperimentConfig,
    paths: RunPaths,
    role: str,
    run_dir: str | Path,
    *,
    popen: Callable[..., "subprocess.Popen[str]"] = subprocess.Popen,
) -> Path:
    """Materialize the verl config for `role` and launch training.

    Returns the materialized `verl_config_{role}.yaml` path on success.
    Raises `TrainingFailedError` (with the last 50 lines of output) if the
    subprocess exits nonzero. `popen` is injectable so callers/tests can
    substitute a fake process instead of actually invoking verl.
    """
    run_dir = Path(run_dir)
    verl_config_path = materialize_verl_config(cfg, paths, role, run_dir)
    resolved_config_path = save_resolved(cfg, run_dir)

    env = {name: os.environ[name] for name in _FORWARDED_ENV_VARS if name in os.environ}
    env["STORAGE_PATH"] = str(paths.storage)
    env["EXPERIMENT_CONFIG_PATH"] = str(resolved_config_path)
    env["VERL_EXPERIMENT_NAME"] = paths.save_name(role)
    env["COEVOLVE_ITERATION"] = str(paths.iteration)
    env.setdefault("WANDB_MODE", "disabled")

    cmd = [sys.executable, "-m", "verl.trainer.main", f"config={verl_config_path}"]

    proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=REPO_ROOT)
    tail: deque[str] = deque(maxlen=_TAIL_LINES)
    for line in proc.stdout:
        print(line, end="")
        tail.append(line)
    returncode = proc.wait()

    if returncode != 0:
        raise TrainingFailedError(
            f"verl.trainer.main exited {returncode} for role={role}:\n" + "".join(tail)
        )

    return verl_config_path
