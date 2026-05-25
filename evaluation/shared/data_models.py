"""
Shared data models for solver sampling pipeline.

WritingPromptResponse: Stores all M sampled responses with evaluation scores.
Used across generate → evaluate → rewards pipeline.

Follows the same RewardScore pattern as verl/workers/reward/function.py:
- overall: Overall uncertainty reward (R-Zero formula)
- format: Format validity (1.0 if valid response, else 0.0)
- accuracy: Mean score across criteria (normalized [0, 1])
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, TypedDict
import json
from statistics import mean, stdev

from .rewards import compute_writing_reward


class RewardScore(TypedDict):
    """Reward score structure (matches verl/workers/reward/function.py)"""
    overall: float  # Overall uncertainty reward (R-Zero formula)
    format: Optional[float]  # Format validity (0.0-1.0)
    accuracy: Optional[float]  # Mean score across criteria (0.0-1.0)


@dataclass
class SampleResult:
    """Single generated response with criterion-level scores."""

    sample_id: int  # 0, 1, 2, ... (M-1)
    text: str  # Generated response text
    scores: Dict[str, Dict[str, Any]]  # {criterion_name: {score, reason}}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WritingPromptResponse:
    """
    Complete sampling result for one prompt.

    Stores M sampled responses and aggregated statistics.
    Used as the unified data structure across the pipeline.

    FORMAT CHECK (from R-Zero paper):
    If format_score == -1 (question fails format validation):
    - No sampling is performed
    - overall reward is set to 0.0
    - No further processing occurs
    """

    # Metadata
    prompt_id: str
    domain: str  # D1-D6
    domain_name: str
    subdomain: str
    query: str
    criteria: List[Dict[str, Any]] = field(default_factory=list)
    format_score: int = 1  # From WritingPrompt: 1 if valid, -1 if invalid format

    # Sampling results
    samples: List[SampleResult] = field(default_factory=list)
    M: int = 0  # Number of samples

    # Aggregated statistics (computed on-demand or set explicitly)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Example structure:
    # {
    #   "Specificity and Detail": {
    #     "mean": 7.2,
    #     "std": 1.1,
    #     "min": 6,
    #     "max": 9,
    #     "percentile_25": 6.5,
    #     "percentile_75": 8.0
    #   }
    # }

    # Reward (computed post-hoc, matches verl/workers/reward/function.py structure)
    reward_score: Optional[RewardScore] = None

    def add_sample(self, sample: SampleResult) -> None:
        """Add a sample response to this prompt."""
        self.samples.append(sample)
        self.M = len(self.samples)

    def is_valid_response(self) -> bool:
        """Check if all samples have complete evaluation data."""
        if not self.samples:
            return False
        for sample in self.samples:
            if not sample.scores:
                return False
        return True

    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregate statistics across M samples for each criterion.

        Returns:
            {criterion_name: {mean, std, min, max, percentile_25, percentile_75}}
        """
        if not self.samples or not self.criteria:
            return {}

        stats = {}

        # For each criterion, collect scores across all M samples
        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            scores = []

            for sample in self.samples:
                if criterion_name in sample.scores:
                    score_entry = sample.scores[criterion_name]
                    # Extract numeric score
                    score = score_entry.get("score", None)
                    if score is not None:
                        scores.append(float(score))

            if scores:
                sorted_scores = sorted(scores)
                stats[criterion_name] = {
                    "mean": mean(scores),
                    "std": stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores),
                    "percentile_25": sorted_scores[len(scores) // 4]
                    if len(scores) >= 4
                    else sorted_scores[0],
                    "percentile_75": sorted_scores[(3 * len(scores)) // 4]
                    if len(scores) >= 4
                    else sorted_scores[-1],
                }

        self.statistics = stats
        return stats

    def compute_aggregate_score(self, aggregate_fn="mean") -> float:
        """
        Compute single aggregate score across all criteria and samples.

        Args:
            aggregate_fn: "mean" or "median"

        Returns:
            Single float score [1-10]
        """
        if not self.statistics:
            self.compute_statistics()

        all_means = [
            stats["mean"] for stats in self.statistics.values()
        ]

        if not all_means:
            return 0.0

        if aggregate_fn == "mean":
            return sum(all_means) / len(all_means)
        elif aggregate_fn == "median":
            sorted_means = sorted(all_means)
            n = len(sorted_means)
            return (
                sorted_means[n // 2]
                if n % 2 == 1
                else (sorted_means[n // 2 - 1] + sorted_means[n // 2]) / 2
            )
        else:
            return sum(all_means) / len(all_means)

    def compute_reward_uncertainty(self) -> RewardScore:
        """
        Compute R-Zero uncertainty reward and other components.

        FORMAT CHECK (from R-Zero paper):
        If format_score == -1 (question fails format validation):
        - overall reward is immediately set to 0.0
        - No solver sampling should have occurred
        - No further processing occurs

        Otherwise:
        r_uncertainty(x; φ) = 1 - 2|p̂(x; S_φ) - 1/2|

        Where p̂(x; S_φ) is the average score normalized to [0,1].

        For WritingBench (1-10 scale):
        - Normalize: mean_score / 10
        - overall: 1 - 2|normalized_score - 0.5| (uncertainty reward) OR 0.0 if format fails
        - format: 1.0 if valid, -1.0 if invalid
        - accuracy: normalized_score (mean score normalized to [0,1]) OR 0.0 if format fails

        Returns:
            RewardScore with overall, format, accuracy components.
        """
        if not self.statistics:
            self.compute_statistics()

        aggregate_score = self.compute_aggregate_score()
        reward_score: RewardScore = compute_writing_reward(
            avg_score=aggregate_score,
            format_valid=(self.format_score != -1),
        )
        self.reward_score = reward_score
        return reward_score

    def get_overall_reward(self) -> float:
        """Get the overall uncertainty reward (convenience method)."""
        if self.reward_score is None:
            self.compute_reward_uncertainty()
        return self.reward_score["overall"]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (JSON-compatible)."""
        return {
            "prompt_id": self.prompt_id,
            "domain": self.domain,
            "domain_name": self.domain_name,
            "subdomain": self.subdomain,
            "query": self.query,
            "criteria": self.criteria,
            "format_score": self.format_score,
            "M": self.M,
            "samples": [s.to_dict() for s in self.samples],
            "statistics": self.statistics,
            "reward_score": self.reward_score,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_jsonl(self) -> str:
        """Serialize to single JSONL line (no indent)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WritingPromptResponse":
        """Deserialize from dictionary."""
        samples = [
            SampleResult(
                sample_id=s["sample_id"],
                text=s["text"],
                scores=s["scores"],
            )
            for s in data.get("samples", [])
        ]

        response = cls(
            prompt_id=data["prompt_id"],
            domain=data.get("domain", ""),
            domain_name=data.get("domain_name", ""),
            subdomain=data.get("subdomain", ""),
            query=data["query"],
            criteria=data.get("criteria", []),
            format_score=data.get("format_score", 1),
            samples=samples,
            M=data.get("M", len(samples)),
        )

        response.statistics = data.get("statistics", {})
        response.reward_score = data.get("reward_score")

        return response
