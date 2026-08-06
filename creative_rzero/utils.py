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

logger = logging.getLogger(__name__)

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
    def validate_output_tags(text: str) -> tuple[bool, str]:
        """Check if output has proper <output>...</output> tags."""
        match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
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
    def validate_response(response: str) -> tuple[int, dict | list | None]:
        """
        Validate complete response format (XML tags + JSON).

        Expects the plain answer — callers must strip any <think>…</think> trace
        (via split_thinking) before calling, so the <output> parser can't match
        tags that appear inside the reasoning.

        Returns:
            Tuple of (score, parsed_json)
            score: 1 if valid, -1 if invalid format
        """
        has_tags, extracted = FormatValidator.validate_output_tags(response)
        if not has_tags:
            logger.warning("Missing <output> tags in response")
            return -1, None

        is_valid_json, parsed = FormatValidator.validate_json(extracted)
        if not is_valid_json:
            logger.warning(f"Invalid JSON in output block: {extracted[:200]}")
            return -1, None

        # Ensure the top-level value is a dict with a string query field.
        # The model can occasionally emit `"query": {...}` instead of a plain
        # string, which would cause callers to crash on .strip().
        if not isinstance(parsed, dict):
            logger.warning(f"Parsed JSON is not a dict: {type(parsed)}")
            return -1, None
        query_val = parsed.get("query")
        if query_val is not None and not isinstance(query_val, str):
            logger.warning(
                f"'query' field is {type(query_val).__name__}, expected str; "
                f"value preview: {str(query_val)[:120]}"
            )
            return -1, None

        if query_val and not is_english_output(query_val):
            logger.warning(f"Non-English characters detected in query: {query_val[:80]!r}")
            return -1, None

        return 1, parsed
