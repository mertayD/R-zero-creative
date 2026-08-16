"""
raw.py — raw judged-score solver reward (T3.4).

`overall` is the judge's own avg criterion score, rescaled to [0, 1]
(avg_score / 10.0), with no within-group ranking. GRPO's own group
mean/std normalisation supplies the zero-centering that rank mode gives
for free — but unlike rank, raw preserves *how much* better one response
is than another, not just its ordinal position.

Unlike RankReward, a low_quality group is NOT overridden to a uniform
value here: a uniformly-low raw score is itself a real, non-noisy signal
(every response actually was bad), whereas a uniformly-low *rank* would be
ranking noise (there's no ordering left to extract). `group.low_quality`
is still computed and available — the caller logs it either way (see
creative_solver_caller.py's `is_low_quality` JSONL field) — this strategy
just doesn't act on it.
"""

from __future__ import annotations

from creative_rzero.rewards.registry import GroupReward, GroupScores, RewardStrategy, register


@register("raw")
class RawReward(RewardStrategy):
    def score_group(self, group: GroupScores) -> GroupReward:
        if group.all_failed:
            return GroupReward(overall={idx: 0.0 for idx in group.scores})

        return GroupReward(
            overall={idx: round(score / 10.0, 4) for idx, score in group.scores.items()}
        )
