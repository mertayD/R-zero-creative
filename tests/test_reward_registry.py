"""Tests for creative_rzero/rewards/registry.py — the T3.3 strategy
registry itself (register/get plumbing), independent of any concrete
strategy's math."""

import pytest

from creative_rzero.rewards.registry import GroupReward, RewardStrategy, get_strategy, register


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Every test gets its own empty registry so fixtures here can't leak
    into each other or collide with the real "rank"/"raw" registrations."""
    monkeypatch.setattr("creative_rzero.rewards.registry._REGISTRY", {})


def test_register_and_get_roundtrip():
    @register("dummy")
    class DummyReward(RewardStrategy):
        def score_group(self, group):
            return GroupReward(overall={idx: 1.0 for idx in group.scores})

    assert get_strategy("dummy") is DummyReward
    assert DummyReward.name == "dummy"


def test_get_unknown_strategy_raises_with_available_names():
    @register("known")
    class KnownReward(RewardStrategy):
        def score_group(self, group):
            return GroupReward(overall={})

    with pytest.raises(KeyError, match="known"):
        get_strategy("unknown")


def test_registering_same_name_twice_with_different_class_raises():
    @register("dupe")
    class First(RewardStrategy):
        def score_group(self, group):
            return GroupReward(overall={})

    with pytest.raises(ValueError, match="already registered"):

        @register("dupe")
        class Second(RewardStrategy):
            def score_group(self, group):
                return GroupReward(overall={})


def test_reregistering_the_identical_class_is_a_noop():
    @register("same")
    class Same(RewardStrategy):
        def score_group(self, group):
            return GroupReward(overall={})

    register("same")(Same)  # e.g. a module reload — must not raise
    assert get_strategy("same") is Same
