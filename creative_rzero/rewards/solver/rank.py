"""
rank.py — normalised-rank solver reward (T3.4).

Within each group, rank responses by judged score and assign a normalised
rank in [0, 1]; mean 0.5 → GRPO advantages are zero-centred by
construction, independent of the judge's absolute scale.

    R_i = (G_eff - rank_i) / (G_eff - 1)    rank_i ∈ {1 … G_eff}  (1 = best)

Ties get the average of the ranks they span (scipy's "average" method), so
equal scores get equal rewards and contribute zero GRPO advantage between
themselves — with plain ordinal ranks, tie-breaking (typically by original
order) injects pure noise, and most G>=2 groups have at least one tie on a
1-10 judge scale.

A low_quality group (scored fine, but every response was mediocre-to-bad)
is deliberately NOT ranked: with no real quality spread to extract,
ranking would just rank noise. It gets the same neutral 0.5 every sample
would get in a genuine tie, instead.
"""

from __future__ import annotations

from typing import Dict

from scipy.stats import rankdata

from creative_rzero.rewards.registry import GroupReward, GroupScores, RewardStrategy, register


@register("rank")
class RankReward(RewardStrategy):
    def score_group(self, group: GroupScores) -> GroupReward:
        if group.all_failed:
            return GroupReward(overall={idx: 0.0 for idx in group.scores})

        if group.low_quality:
            return GroupReward(overall={idx: 0.5 for idx in group.scores})

        return GroupReward(overall=_normalised_rank(group.scores))


def _normalised_rank(scores: Dict[int, float]) -> Dict[int, float]:
    """Convert {sample_idx: avg_score} into normalised rank rewards.

    G_eff = len(scores) so the formula adapts when some rollouts fail to
    generate (M requested -> M-1 produced).

    G_eff == 1 -> {idx: 0.5}  (neutral, no signal, avoids division by zero).
    All-equal group -> every sample gets the average rank -> all 0.5.
    """
    G_eff = len(scores)
    if G_eff == 0:
        return {}
    if G_eff == 1:
        idx = next(iter(scores))
        return {idx: 0.5}

    sample_idxs = list(scores.keys())
    values = [scores[idx] for idx in sample_idxs]
    # Negate so the highest score gets rank 1 (rankdata ranks ascending).
    ranks = rankdata([-v for v in values], method="average")

    rewards: Dict[int, float] = {}
    for sample_idx, rank in zip(sample_idxs, ranks):
        rewards[sample_idx] = round(float((G_eff - rank) / (G_eff - 1)), 4)
    return rewards
