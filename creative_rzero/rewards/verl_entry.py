"""verl_entry.py — the ONE file verl's `worker.reward.reward_function` points at.

`configs/base.yaml` has pointed here since T2.1, but the file didn't exist —
any run of the new pipeline crashed (or, worse, silently degraded: verl's
`RewardConfig.post_init` sets `reward_function = None` when the path doesn't
exist) the moment the reward manager loaded. This is the minimal T3.6
adapter that makes T2.8/T2.9 runnable: it dispatches by role to the two
existing reward callers. The Phase-3 registry (T3.3) replaces the dispatch
dict with `rewards/registry.py` lookups without changing this file's
contract.

Contract with `steps/train_verl.py`:
  * `role` arrives as a `reward_function_kwargs` entry in the materialized
    verl config ("role passed via config, not filename" — REFACTOR_PLAN
    T3.6), applied by verl via `functools.partial`.
  * `EXPERIMENT_CONFIG_PATH` (set in the training subprocess env) points at
    the resolved experiment config, the source of truth for every reward
    knob.

The legacy callers read their knobs from ambient env vars at import time
(pre-refactor bash exported them; nothing does anymore — exactly the
class-A silent-default bug). Until Phase 3 rewrites them as strategy
classes taking injected config, `_bridge_config_to_env` projects the
resolved config into those env names *before* the caller module is
imported, so every value the reward path uses traces back to the one
config file.

verl loads this file standalone via `spec_from_file_location` (module name
`custom_reward_fn`, no package context), so sys.path is pinned from
`__file__` — not from `REMOTE_REPO_PATH` — before any repo import.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# The callers import `batch_eval_agent` flat (not as evaluation.writing_bench.*).
_WB_DIR = _REPO_ROOT / "evaluation" / "writing_bench"
for _p in (str(_REPO_ROOT), str(_WB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VALID_ROLES = ("challenger", "solver")

# role -> (module path, the strategy names that module implements today).
# The strategy name is checked against `rewards.<role>.type` so a config
# asking for a strategy these legacy callers don't implement fails loudly
# instead of silently computing the wrong reward. creative_solver_caller
# implements both "rank" and "raw" internally, switched by the
# CREATIVE_SOLVER_REWARD_TYPE env var bridged below.
# TODO(dayanc): replace this dict with a registry lookup once the Phase-3 registry is in place.
_ROLE_IMPLS = {
    "solver": ("examples.reward_function.creative_solver_caller", ("rank", "raw")),
    "challenger": ("examples.reward_function.creative_writing_caller", ("uncertainty",)),
}

_impl_cache: Dict[str, Callable] = {}


def _load_experiment_config():
    cfg_path = os.environ.get("EXPERIMENT_CONFIG_PATH")
    if not cfg_path:
        raise RuntimeError(
            "EXPERIMENT_CONFIG_PATH is not set — verl_entry.py needs the resolved "
            "experiment config to configure the reward path. steps/train_verl.py "
            "sets it; if you're launching verl by hand, export it first."
        )
    if not os.path.exists(cfg_path):
        raise RuntimeError(f"EXPERIMENT_CONFIG_PATH={cfg_path} does not exist")
    from omegaconf import OmegaConf

    return OmegaConf.load(cfg_path)


def _bridge_config_to_env(cfg) -> None:
    """Project resolved-config values into the env names the legacy reward
    callers and judge client read at import time. Direct assignment, not
    setdefault: the resolved config is the source of truth, ambient env is
    not."""
    bridge = {
        # judge client (evaluation/writing_bench/evaluator/llm.py / mock.py / critic_server.py)
        "WB_JUDGE_TYPE": cfg.judge.type,
        "WB_JUDGE_MODEL": cfg.judge.model,
        "WB_JUDGE_MAX_TOKENS": cfg.judge.max_tokens,
        "JUDGE_MAX_HTTP_RETRY_ATTEMPTS": cfg.judge.http_max_retry_attempts,
        # sft-critic backend only (evaluator/critic_server.py::CriticServerAgent)
        "WB_CRITIC_URL": cfg.judge.critic_url or "",
        "WB_CRITIC_MODEL": cfg.judge.critic_model,
        # both callers
        "CREATIVE_SCORER_MAX_WORKERS": cfg.judge.max_workers,
        "TRUNCATION_MARGIN_TOKENS": cfg.judge.truncation_margin_tokens,
        # solver caller (creative_solver_caller.py)
        "SOLVER_MAX_RESPONSE_LENGTH": cfg.solver.max_response_length,
        "CREATIVE_LOW_QUALITY_THRESHOLD": cfg.judge.low_quality_threshold,
        "CREATIVE_SOLVER_REWARD_TYPE": cfg.rewards.solver.type,
        # challenger caller (creative_writing_caller.py)
        "CHALLENGER_MAX_RESPONSE_LENGTH": cfg.challenger.max_response_length,
        "CREATIVE_SOLVER_PORT": cfg.challenger.solver_query.port,
        "CREATIVE_SOLVER_MAX_TOKENS": cfg.challenger.solver_query.max_tokens,
        # R-Diverse penalties (rewards/rdiverse_penalty.py)
        "CHALLENGER_PENALTY_ENABLED": "1" if cfg.rewards.penalty.enabled else "0",
        "CHALLENGER_PENALTY_ALPHA": cfg.rewards.penalty.alpha,
        "CHALLENGER_PENALTY_BETA": cfg.rewards.penalty.beta,
        "CHALLENGER_PENALTY_LAMBDA": cfg.rewards.penalty.lam,
        "CHALLENGER_PENALTY_TAU_MAX": cfg.rewards.penalty.tau_max,
        "CHALLENGER_PENALTY_TAU_MEAN": cfg.rewards.penalty.tau_mean,
        "CHALLENGER_PENALTY_CLUSTER_T": cfg.rewards.penalty.cluster_threshold,
        "CHALLENGER_PENALTY_EMBED_MODEL": cfg.rewards.penalty.embed_model,
        "CHALLENGER_PENALTY_MEMORY_UPDATE": cfg.rewards.penalty.memory_update,
    }
    os.environ.update({k: str(v) for k, v in bridge.items()})


def _get_impl(role: str) -> Callable:
    """Load (once per process) the reward implementation for `role`."""
    if role in _impl_cache:
        return _impl_cache[role]
    if role not in VALID_ROLES:
        raise ValueError(
            f"role={role!r} must be one of {VALID_ROLES} — steps/train_verl.py passes it "
            "via worker.reward.reward_function_kwargs in the materialized verl config."
        )

    cfg = _load_experiment_config()
    module_path, implemented_strategies = _ROLE_IMPLS[role]
    configured_strategy = cfg.rewards[role].type
    if configured_strategy not in implemented_strategies:
        raise ValueError(
            f"rewards.{role}.type={configured_strategy!r} is not available yet — the "
            f"pre-registry caller behind verl_entry implements only {implemented_strategies!r} "
            "(the T3.3/T3.4/T3.5 registry adds the rest)."
        )

    _bridge_config_to_env(cfg)  # must precede the import below (import-time env reads)
    module = importlib.import_module(module_path)
    _impl_cache[role] = module.compute_score
    return _impl_cache[role]


def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    role: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Batch reward entrypoint (verl `reward_type: batch` signature, plus the
    `role` kwarg injected from `reward_function_kwargs`)."""
    if role is None:
        raise ValueError(
            "verl_entry.compute_score called without a role — the materialized verl "
            "config must set worker.reward.reward_function_kwargs: {role: challenger|solver} "
            "(steps/train_verl.py does this)."
        )
    return _get_impl(role)(predicts, ground_truths)
