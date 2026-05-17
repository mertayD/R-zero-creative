"""
BatchEvalAgent — scores all criteria for a single response in one Claude call.

Replaces the per-criterion loop in evaluate_benchmark.py with a single API
call that returns a JSON object keyed by criterion name, reducing API calls
from N_criteria per sample down to 1.

Usage:
    from evaluator import ClaudeAgent
    from batch_eval_agent import BatchEvalAgent
    from batch_eval_prompt import batch_evaluate_system

    agent = BatchEvalAgent(ClaudeAgent(system_prompt=batch_evaluate_system))
    scores = agent.score_all_criteria(
        content={"response": "..."},
        query="...",
        criteria=[{"name": "...", "criteria_description": "...", "1-2": "...", ...}, ...],
    )
    # scores == {"Criterion A": {"score": 8, "reason": "..."}, ...}
"""

import json

from batch_eval_prompt import batch_evaluate_prompt
from evaluate_benchmark import process_gen_field


def _format_criterion(c: dict) -> str:
    lines = [f"Name: {c['name']}", f"Description: {c['criteria_description']}"]
    for key in ("1-2", "3-4", "5-6", "7-8", "9-10"):
        if key in c:
            lines.append(f"  {key}: {c[key]}")
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) if present."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            if "\n" in inner:
                inner = inner.split("\n", 1)[1]
            return inner.strip()
    return text


class BatchEvalAgent:
    def __init__(self, agent):
        self.agent = agent

    def success_check_fn(self, response: str, expected_names: set) -> bool:
        try:
            result = json.loads(_strip_fences(response))
        except (json.JSONDecodeError, TypeError):
            return False
        valid_scores = set(range(1, 11))
        return all(
            name in result
            and isinstance(result[name].get("score"), int)
            and result[name]["score"] in valid_scores
            for name in expected_names
        )

    def score_all_criteria(
        self,
        content: dict,
        query: str,
        criteria: list,
        max_retries: int = 3,
    ) -> dict:
        """
        Score all criteria for one response in a single API call.

        Args:
            content:     Dict with a "response" key (same convention as EvalAgent).
            query:       The writing prompt query string.
            criteria:    List of criterion dicts with "name", "criteria_description",
                         and optional "1-2" / "3-4" / ... rubric fields.
            max_retries: Number of retry attempts on parse or validation failure.

        Returns:
            Dict mapping criterion name -> {"score": int, "reason": str}.

        Raises:
            ValueError if scoring fails after max_retries attempts.
        """
        criteria_block = "\n\n".join(_format_criterion(c) for c in criteria)
        expected_names = {c["name"] for c in criteria}

        prompt = batch_evaluate_prompt.format(
            criteria_block=criteria_block,
            query=query,
            response=process_gen_field(content["response"]),
        )

        retry = 0
        response = None
        while retry < max_retries:
            response, success = self.agent.run(
                prompt=prompt,
                success_check_fn=lambda r: self.success_check_fn(r, expected_names),
            )
            if success:
                try:
                    return json.loads(_strip_fences(response))
                except (json.JSONDecodeError, TypeError):
                    pass
            retry += 1

        raise ValueError(
            f"Failed to score all criteria after {max_retries} attempts. "
            f"Last response: {response!r}"
        )
