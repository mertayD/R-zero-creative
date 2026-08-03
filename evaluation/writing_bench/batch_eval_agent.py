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
from evaluator.llm import JudgeAPIError


class JudgeParseError(Exception):
    """Judge responded successfully every attempt, but the response never
    parsed/validated as the expected per-criterion JSON — distinct from a
    JudgeAPIError, which means the judge call itself failed."""


class JudgeInputError(Exception):
    """A criterion dict was missing a required field before any judge call
    was made — a caller/data bug (e.g. a malformed WritingPrompt.criteria
    entry), not a judge failure. Raised eagerly so it fails at the same
    layer as the other two, instead of surfacing as a bare KeyError deep
    inside _format_criterion."""


def _format_criterion(c: dict) -> str:
    try:
        name = c["name"]
        description = c["criteria_description"]
    except KeyError as e:
        raise JudgeInputError(f"criterion missing required field {e}: {c!r}") from e
    lines = [f"Name: {name}", f"Description: {description}"]
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
        """Predicate only — must never raise, so a malformed-but-parseable
        response (e.g. {"Criterion A": 8} instead of {"Criterion A":
        {"score": 8}}) is treated as "didn't validate, retry" rather than
        crashing run()'s retry loop with an unhandled AttributeError/KeyError.
        Exhausting retries on a persistently malformed shape still concludes
        cleanly: score_all_criteria raises JudgeParseError, a typed,
        catchable outcome instead of a stray crash three layers up."""
        try:
            result = json.loads(_strip_fences(response))
            valid_scores = set(range(1, 11))
            return all(
                name in result
                and isinstance(result[name].get("score"), int)
                and result[name]["score"] in valid_scores
                for name in expected_names
            )
        except (json.JSONDecodeError, TypeError, AttributeError, KeyError):
            return False

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
            max_retries: Retry budget for a judge call that responds but fails
                         validation — forwarded directly as ClaudeAgent.run's
                         max_try. There is deliberately only one retry loop:
                         run() already retries against this exact predicate
                         (success_check_fn=self.success_check_fn below), so a
                         second loop here would only silently multiply the
                         retry budget (max_retries * run()'s own max_try)
                         while retrying the identical check.

        Returns:
            Dict mapping criterion name -> {"score": int, "reason": str}.

        Raises:
            JudgeInputError if a criterion dict is missing a required field
                           — fails before any judge call is made.
            JudgeAPIError  if the judge call itself fails (network/HTTP,
                           after call_claude's own retry budget is exhausted)
                           — propagates immediately, no parse-retry attempted.
            JudgeParseError if the judge responds every attempt but the
                           response never parses/validates after max_retries.
        """
        criteria_block = "\n\n".join(_format_criterion(c) for c in criteria)
        expected_names = {c["name"] for c in criteria}

        prompt = batch_evaluate_prompt.format(
            criteria_block=criteria_block,
            query=query,
            response=process_gen_field(content["response"]),
        )

        # A JudgeAPIError means call_claude already exhausted its own retry
        # budget on network/HTTP failures — it propagates straight out of
        # run(), uncaught here, rather than being treated as a validation
        # failure worth retrying.
        response, success = self.agent.run(
            prompt=prompt,
            max_try=max_retries,
            success_check_fn=lambda r: self.success_check_fn(r, expected_names),
        )
        if not success:
            raise JudgeParseError(
                f"Failed to score all criteria after {max_retries} attempts. "
                f"Last response: {response!r}"
            )

        # success_check_fn already proved this parses (same input, same
        # deterministic _strip_fences) — no try/except here so a future
        # divergence between the two would surface loudly, not get swallowed.
        return json.loads(_strip_fences(response))
