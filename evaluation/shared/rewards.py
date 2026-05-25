"""R-Zero uncertainty reward formula for WritingBench creative writing evaluation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_models import RewardScore


def r_zero_uncertainty(avg_score: float, max_score: float = 10.0) -> float:
    """
    Compute R-Zero uncertainty reward from a mean criterion score.

    r_uncertainty(x; φ) = 1 - 2 * |p̂(x; S_φ) - 0.5|

    where p̂ = avg_score / max_score normalises the score to [0, 1].

    Returns 1.0 at the midpoint (maximum uncertainty) and approaches 0.0 at
    both extremes (solver is always right or always wrong).

    Args:
        avg_score: Mean criterion score (e.g. 0–10).
        max_score: Upper bound of the score scale (default 10.0).

    Returns:
        Uncertainty reward in [0.0, 1.0].
    """
    accuracy = avg_score / max_score
    return 1.0 - 2.0 * abs(accuracy - 0.5)


def compute_writing_reward(avg_score: float, format_valid: bool, max_score: float = 10.0) -> "RewardScore":
    """
    Compute the full R-Zero RewardScore for a WritingBench response.

    Args:
        avg_score:    Mean criterion score across all criteria and samples.
        format_valid: Whether the generated prompt passed format validation.
        max_score:    Upper bound of the score scale (default 10.0).

    Returns:
        RewardScore dict with overall, format, accuracy keys.
    """
    if not format_valid:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    accuracy = avg_score / max_score
    overall  = r_zero_uncertainty(avg_score, max_score)
    return {
        "overall":  round(overall, 4),
        "format":   1.0,
        "accuracy": round(accuracy, 4),
    }
