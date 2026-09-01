"""creative_rzero/utils.py — text/format validation utilities.

Shared by prompt generation (`creative_rzero/steps/generate_prompts.py`,
which validates its own model output before accepting a generated prompt)
and the reward callers that grade rollouts against that output
(`examples/reward_function/creative_*_caller.py`). Kept dependency-light
(regex + json only) so every side can import it without pulling in
vllm/transformers.
"""

from __future__ import annotations

import json
import logging
from typing import List

import regex as re

from creative_rzero.failure_reasons import FormatFailureReason

logger = logging.getLogger(__name__)

# Forced assistant prefix for challenger training rollouts: generation starts
# inside the fenced JSON, which removes the free-form opening tokens where
# 4B-Base degenerates (garbage prefixes / role-token leaks / prompt echo).
# The rollout prompt appends it (data.response_prefill) and the reward caller
# prepends it back before validation, so parsed text is a complete document.
CHALLENGER_PREFILL = "```json\n{"

_CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿\U00020000-\U0002a6df]')


def is_english_output(text: str) -> bool:
    """Return False if text contains CJK (Chinese/Japanese/Korean) characters."""
    return not bool(_CJK_RE.search(text))


def find_cjk_matches(text: str) -> List[str]:
    """Return every CJK (Chinese/Japanese/Korean) character found in text.

    Callers that filter on this (e.g. the solver reward caller's language
    filter) use it to count/preview what actually triggered the filter,
    instead of only knowing pass/fail from is_english_output.
    """
    return _CJK_RE.findall(text)


class FormatValidator:
    """Validates model output format and provides scoring."""

    @staticmethod
    def validate_json_fence(text: str) -> tuple[bool, str]:
        """Extract the first ```json ... ``` fenced block (bare ``` fences
        accepted too — models drop the language tag often enough that
        rejecting it would only add noise to the failure counts)."""
        match = re.search(r'```(?:\s*json)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if match:
            return True, match.group(1).strip()
        return False, ""

    @staticmethod
    def validate_json(text: str) -> tuple[bool, dict | list | None]:
        """Check if text is valid JSON."""
        try:
            parsed = json.loads(text)
            return True, parsed
        except (json.JSONDecodeError, ValueError):
            return False, None

    @staticmethod
    def validate_response(response: str) -> tuple[int, dict | list | None, str]:
        """
        Validate complete response format (XML tags + JSON).

        Expects the plain answer — callers must strip any <think>…</think> trace
        (via split_thinking) before calling, so the fence parser can't match
        a code block that appears inside the reasoning.

        Returns:
            Tuple of (score, parsed_json, failure_reason)
            score: 1 if valid, -1 if invalid format
            failure_reason: a `FormatFailureReason` value — "ok" when valid,
                otherwise which check failed. Callers aggregate these per run
                to see where format failures concentrate.
        """
        has_fence, extracted = FormatValidator.validate_json_fence(response)
        if not has_fence:
            logger.warning("Missing ```json fence in response")
            return -1, None, FormatFailureReason.MISSING_JSON_FENCE.value

        is_valid_json, parsed = FormatValidator.validate_json(extracted)
        if not is_valid_json:
            logger.warning(f"Invalid JSON in output block: {extracted[:200]}")
            return -1, None, FormatFailureReason.INVALID_JSON.value

        # Ensure the top-level value is a dict with a string query field.
        # The model can occasionally emit `"query": {...}` instead of a plain
        # string, which would cause callers to crash on .strip().
        if not isinstance(parsed, dict):
            logger.warning(f"Parsed JSON is not a dict: {type(parsed)}")
            return -1, None, FormatFailureReason.TOP_LEVEL_NOT_DICT.value
        query_val = parsed.get("query")
        if query_val is not None and not isinstance(query_val, str):
            logger.warning(
                f"'query' field is {type(query_val).__name__}, expected str; "
                f"value preview: {str(query_val)[:120]}"
            )
            return -1, None, FormatFailureReason.QUERY_NOT_STRING.value

        if query_val and not is_english_output(query_val):
            logger.warning(f"Non-English characters detected in query: {query_val[:80]!r}")
            return -1, None, FormatFailureReason.NON_ENGLISH_QUERY.value

        return 1, parsed, FormatFailureReason.OK.value
