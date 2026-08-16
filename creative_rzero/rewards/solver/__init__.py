"""Solver reward strategies — importing this package registers all of
them (rank, raw, ...) with creative_rzero.rewards.registry."""

from creative_rzero.rewards.solver.rank import RankReward
from creative_rzero.rewards.solver.raw import RawReward

__all__ = ["RankReward", "RawReward"]
