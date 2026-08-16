"""Unit tests for the solver reward strategies (T3.4) — pure math over
GroupScores fixtures. No judge, no network, no env vars: this is exactly
what "self-contained reward strategy" (T3.3) is meant to buy — a new
formula is testable in isolation from creative_solver_caller.py."""

import pytest

import creative_rzero.rewards.solver  # noqa: F401 — registers rank/raw
from creative_rzero.rewards.registry import GroupScores, get_strategy

rank = get_strategy("rank")()
raw = get_strategy("raw")()


class TestRankReward:
    def test_all_failed_group_gets_uniform_zero(self):
        group = GroupScores(scores={0: 0.0, 1: 0.0}, all_failed=True, low_quality=False)
        assert rank.score_group(group).overall == {0: 0.0, 1: 0.0}

    def test_low_quality_group_gets_uniform_half(self):
        group = GroupScores(scores={0: 1.5, 1: 2.0}, all_failed=False, low_quality=True)
        assert rank.score_group(group).overall == {0: 0.5, 1: 0.5}

    def test_g_eff_one_is_neutral(self):
        group = GroupScores(scores={5: 7.0}, all_failed=False, low_quality=False)
        assert rank.score_group(group).overall == {5: 0.5}

    def test_best_and_worst_get_the_extremes(self):
        group = GroupScores(scores={0: 9.0, 1: 5.0, 2: 1.0}, all_failed=False, low_quality=False)
        overall = rank.score_group(group).overall
        assert overall[0] == 1.0
        assert overall[2] == 0.0
        assert overall[0] > overall[1] > overall[2]

    def test_ties_get_the_average_of_their_ranks(self):
        group = GroupScores(scores={0: 8.0, 1: 8.0, 2: 2.0}, all_failed=False, low_quality=False)
        overall = rank.score_group(group).overall
        assert overall[0] == overall[1] == pytest.approx(0.75)  # ranks {1,2} average to 1.5
        assert overall[2] == 0.0

    def test_mean_reward_is_centred_at_half(self):
        group = GroupScores(
            scores={0: 9.0, 1: 6.0, 2: 3.0, 3: 1.0}, all_failed=False, low_quality=False
        )
        overall = rank.score_group(group).overall
        assert sum(overall.values()) / len(overall) == pytest.approx(0.5)


class TestRawReward:
    def test_all_failed_group_gets_uniform_zero(self):
        group = GroupScores(scores={0: 0.0, 1: 0.0}, all_failed=True, low_quality=False)
        assert raw.score_group(group).overall == {0: 0.0, 1: 0.0}

    def test_low_quality_group_is_not_overridden(self):
        """The point of raw mode: a uniformly-low group keeps its real
        (low) score instead of collapsing to rank mode's neutral 0.5."""
        group = GroupScores(scores={0: 1.5, 1: 2.0}, all_failed=False, low_quality=True)
        assert raw.score_group(group).overall == {0: 0.15, 1: 0.2}

    def test_scores_are_rescaled_to_the_unit_interval(self):
        group = GroupScores(scores={0: 10.0, 1: 5.0, 2: 0.1}, all_failed=False, low_quality=False)
        assert raw.score_group(group).overall == {0: 1.0, 1: 0.5, 2: 0.01}

    def test_single_sample_group_is_not_forced_neutral(self):
        """Unlike rank's G_eff==1 special case, raw has no group-size
        dependence — one sample just gets its own rescaled score."""
        group = GroupScores(scores={5: 7.0}, all_failed=False, low_quality=False)
        assert raw.score_group(group).overall == {5: 0.7}
