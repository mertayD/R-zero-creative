"""creative_rzero/eval/challenger/challenger_judge_agent.py — judge transport
for the challenger eval harness.

Reuses evaluation/writing_bench/evaluator/{llm,mock}.py's `.run(prompt) ->
(response, success)` agents as the judge *transport* only; the harness's own
judge_prompts.py supplies a prompt/schema (`domain_adherence`,
`guidance_adherence`, `criteria_quality`) neither agent was originally built
for. `CriticServerAgent` (judge.type=sft-critic elsewhere in the repo) is
deliberately not wired in here — it's a model fine-tuned to score a *solver
response* against *criteria*, not to judge whether a *generated prompt*
fits its assigned domain/guidance; a different task it was never trained on.

MockJudgeAgent caveat: its `run()` (evaluator/mock.py) looks for "Name: X"
lines in the prompt — the shape BatchEvalAgent bakes in for WritingBench's
per-criterion response scoring — and only falls back to a single
`{"score", "reason"}` object when none are found. Every prompt this harness
sends has no such lines, so `--judge-type mock` always hits that fallback,
never the harness's real 3-field schema. `_parse` below detects that shape
and duplicates the single score into all three fields, tagged
`judge_backend="mock"` so it's never confused with a real per-dimension
judgement. `--judge-type mock` is a smoke-test path only (proves the
generate -> judge -> parse -> aggregate plumbing doesn't crash) — real
evaluation always needs `--judge-type claude`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WB_DIR = _REPO_ROOT / "evaluation" / "writing_bench"
for _p in (str(_REPO_ROOT), str(_WB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluator.llm import ClaudeAgent  # noqa: E402
from evaluator.mock import MockJudgeAgent  # noqa: E402

from creative_rzero.eval.challenger import judge_prompts  # noqa: E402

VALID_JUDGE_TYPES = ("claude", "mock")

_EXPECTED_FIELDS = ("domain_adherence", "guidance_adherence", "criteria_quality")


class JudgeParseError(Exception):
    """The judge responded every retry attempt but its output never
    validated as either the harness's 3-field schema or MockJudgeAgent's
    {"score", "reason"} fallback shape."""


def get_agent(judge_type: str):
    """Construct the judge agent for `judge_type`, wired with this harness's
    own system prompt (not WritingBench's `batch_evaluate_system`)."""
    if judge_type == "claude":
        return ClaudeAgent(system_prompt=judge_prompts.SYSTEM_PROMPT)
    if judge_type == "mock":
        return MockJudgeAgent(system_prompt=judge_prompts.SYSTEM_PROMPT)
    raise ValueError(f"judge_type={judge_type!r} must be one of {VALID_JUDGE_TYPES}")


def _strip_fences(text: str) -> str:
    """Remove a ```json ... ``` (or bare ```) wrapper if present. Small,
    deliberately duplicated from batch_eval_agent.py's private helper rather
    than imported — this harness is meant to stand alone from WritingBench's
    internals, and the function is eight lines with no logic worth sharing
    across a package boundary."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            if "\n" in inner:
                inner = inner.split("\n", 1)[1]
            return inner.strip()
    return text


def _parse(response: str) -> dict | None:
    """Parse a judge response into `{domain_adherence, guidance_adherence,
    criteria_quality, reasoning, judge_backend}` — the three dimension
    fields are numeric 1-5, converted from the judge's categorical label
    (judge_prompts.SCORE_LABELS) via judge_prompts.SCORE_LABEL_TO_VALUE.
    Returns None if the response validates as neither the harness's labeled
    schema nor MockJudgeAgent's fallback shape."""
    try:
        result = json.loads(_strip_fences(response))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(result, dict):
        return None

    if all(
        isinstance(result.get(k), str) and result[k] in judge_prompts.SCORE_LABEL_TO_VALUE
        for k in _EXPECTED_FIELDS
    ):
        return {
            "domain_adherence": judge_prompts.SCORE_LABEL_TO_VALUE[result["domain_adherence"]],
            "guidance_adherence": judge_prompts.SCORE_LABEL_TO_VALUE[result["guidance_adherence"]],
            "criteria_quality": judge_prompts.SCORE_LABEL_TO_VALUE[result["criteria_quality"]],
            "reasoning": result.get("reasoning", ""),
            "judge_backend": "real",
        }

    # MockJudgeAgent's fallback {"score" 1-10, "reason"} shape (see module
    # docstring) — halve+round onto our 1-5 scale and duplicate across all
    # three fields; there is no real per-dimension signal to recover here.
    if isinstance(result.get("score"), int) and 1 <= result["score"] <= 10:
        mock_score = min(5, max(1, round(result["score"] / 2)))
        return {
            "domain_adherence": mock_score,
            "guidance_adherence": mock_score,
            "criteria_quality": mock_score,
            "reasoning": result.get("reason", ""),
            "judge_backend": "mock",
        }
    return None


def score_generated_prompt(
    agent,
    *,
    domain_name: str,
    subdomain: str,
    guidance_applied: list[str],
    query: str,
    criteria: list[dict],
    max_retries: int = 3,
) -> dict:
    """Score one format-valid generated prompt. Raises `JudgeParseError` if
    the judge never returns a parseable response within `max_retries`
    (mirrors `BatchEvalAgent.score_all_criteria`'s contract); a
    `JudgeAPIError` from a live Claude call propagates uncaught, same as
    elsewhere in the repo."""
    prompt = judge_prompts.render_judge_prompt(
        domain_name=domain_name,
        subdomain=subdomain,
        guidance_applied=guidance_applied,
        query=query,
        criteria=criteria,
    )
    response, success = agent.run(
        prompt=prompt,
        max_try=max_retries,
        success_check_fn=lambda r: _parse(r) is not None,
    )
    if not success:
        raise JudgeParseError(
            f"judge response never validated after {max_retries} attempts: {response!r}"
        )
    return _parse(response)
