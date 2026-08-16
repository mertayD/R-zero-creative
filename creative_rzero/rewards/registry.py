"""
creative_rzero/rewards/registry.py — T3.3 reward-strategy registry.

A reward strategy owns exactly one decision: given one prompt group's
judged criterion scores, what per-sample `overall` reward does GRPO see?
Everything else a reward caller needs — the judge client, JSONL/W&B
logging, language/truncation filters, the all_failed/low_quality gate
*definitions* — is shared orchestration and stays in the caller (e.g.
examples/reward_function/creative_solver_caller.py). Only the reward
*formula* is pluggable here, so a new hypothesis is a new ~30-line
strategy file plus one line in an experiment config
(`rewards.solver.type: zscore`), not a new branch threaded through an
existing function.

Usage:
    @register("rank")
    class RankReward(RewardStrategy):
        def score_group(self, group: GroupScores) -> GroupReward:
            ...

    strategy_cls = get_strategy("rank")   # raises KeyError (available names) if unregistered
    reward = strategy_cls().score_group(group)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type


@dataclass(frozen=True)
class GroupScores:
    """One prompt group's judged scores — a strategy's only input.

    `scores` holds every scoreable sample in the group (language-filtered /
    unjudged samples are excluded upstream, before any strategy sees them),
    keyed by its index into the batch's flat predicts/rewards list, valued
    by the judge's avg criterion score on a 1-10 scale.

    `all_failed` / `low_quality` are computed once by the caller from one
    shared definition, then handed to every strategy — deliberately:
    strategies are free to react differently to a low-quality group (see
    RankReward vs RawReward), but they can never *disagree* about which
    groups count as low-quality. That drift — the same flag meaning two
    different things depending on which branch of one function you were
    reading — is exactly what made the old rank/raw code hard to follow.
    """

    scores: Dict[int, float]
    all_failed: bool
    low_quality: bool


@dataclass(frozen=True)
class GroupReward:
    """A strategy's output for one group: the `overall` GRPO reward per
    sample index. Must cover every key in the input GroupScores.scores."""

    overall: Dict[int, float]


class RewardStrategy(ABC):
    """One reward formula, self-contained: given a group's judged scores,
    decide the per-sample `overall` reward. No judge calls, no I/O, no
    env/config reads — a strategy must be fully testable against a plain
    GroupScores fixture, with nothing to mock."""

    name: str  # set by @register

    @abstractmethod
    def score_group(self, group: GroupScores) -> GroupReward: ...


_REGISTRY: Dict[str, Type["RewardStrategy"]] = {}


def register(name: str):
    """Class decorator: @register("rank") makes a strategy reachable by
    that name from get_strategy(), and therefore from an experiment
    config's `rewards.<role>.type`."""

    def _decorate(cls: Type[RewardStrategy]) -> Type[RewardStrategy]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"reward strategy {name!r} is already registered to {_REGISTRY[name]!r} "
                f"— refusing to overwrite it with {cls!r}"
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _decorate


def get_strategy(name: str) -> Type[RewardStrategy]:
    """Look up a registered strategy class by config name (e.g.
    cfg.rewards.solver.type). Raises KeyError listing available names if
    `name` isn't registered — callers own turning that into a
    user-pointed error (see creative_solver_caller.py::_get_strategy)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no reward strategy registered as {name!r} — available: {sorted(_REGISTRY)}. "
            "Strategies self-register via @register(...) at import time; make sure the "
            "strategy module is imported before get_strategy() is called."
        ) from None
