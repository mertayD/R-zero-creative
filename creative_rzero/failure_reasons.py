"""creative_rzero/failure_reasons.py — canonical `failure_reason` taxonomy.

T1.4 established this taxonomy, but only as ad hoc string literals: each of
`examples/reward_function/{creative_solver_caller,creative_writing_caller}.py`
carries its own comment-only copy of the list and assigns the values inline
(`failure_reasons[i] = "language_filter"`, etc.) — nothing type-checks a
caller against a typo'd or half-renamed reason. This module is the one
canonical source; new code (starting with `steps/merge_ckpt.py`'s
infra/policy collapse split) should check membership against it instead of
re-declaring string sets.

The two legacy callers still assign plain string literals rather than these
enum members — they're slated for porting into `rewards/` strategy classes
in Phase 3 (T3.4-T3.6), so retrofitting them here would touch code that's
about to be rewritten anyway. `FailureReason` is a `str` subclass, so a
plain string from either caller's JSONL output still compares and hashes
equal to the matching enum member (`"ok" == FailureReason.OK` and
`"ok" in {FailureReason.OK}` both hold) — consumers don't need the
producers to switch over first.
"""

from __future__ import annotations

from enum import Enum


class FailureReason(str, Enum):
    OK = "ok"

    # Solver taxonomy (creative_solver_caller.py)
    EMPTY_ANSWER = "empty_answer"
    INVALID_CRITERIA = "invalid_criteria"
    LANGUAGE_FILTER = "language_filter"

    # Challenger taxonomy (creative_writing_caller.py)
    TRUNCATED = "truncated"
    CHALLENGER_FORMAT_INVALID = "challenger_format_invalid"
    SOLVER_API_ERROR = "solver_api_error"

    # Shared judge-call outcomes (both callers)
    JUDGE_PARSE_FAIL = "judge_parse_fail"
    JUDGE_RATE_LIMIT = "judge_rate_limit"
    JUDGE_API_ERROR = "judge_api_error"


# Judge/API infrastructure trouble, not the policy's own output — zero
# reward from either cause is zero GRPO advantage, but only a policy failure
# (every other non-OK reason) says anything about the checkpoint itself.
INFRA_FAILURE_REASONS = frozenset({FailureReason.JUDGE_API_ERROR, FailureReason.JUDGE_RATE_LIMIT})


class FormatFailureReason(str, Enum):
    """Where exactly parsing/validation of a challenger response's ```json
    fenced block gave up — finer-grained than `FailureReason` (whose
    challenger side collapses every parse problem into
    `challenger_format_invalid`).

    Produced by `creative_rzero.utils.FormatValidator.validate_response`
    (the first five), `steps/generate_prompts.py::validate_one_shot_response`
    (the stricter one-shot checks), and the challenger reward caller's own
    post-parse checks (the last two). Aggregated per run in
    `generation_log["failure_reason_counts"]` and per rollout in the
    challenger JSONL's `format_reason` column. Like `FailureReason`, this is
    a `str` subclass so logged strings compare equal to members; producers
    assign `.value` at serialization boundaries.
    """

    OK = "ok"

    # FormatValidator.validate_response (shared by training + generation)
    MISSING_JSON_FENCE = "missing_json_fence"
    INVALID_JSON = "invalid_json"
    TOP_LEVEL_NOT_DICT = "top_level_not_dict"
    QUERY_NOT_STRING = "query_not_string"
    NON_ENGLISH_QUERY = "non_english_query"

    # validate_one_shot_response's stricter one-shot schema checks
    MISSING_QUERY_OR_CRITERIA = "missing_query_or_criteria"
    CRITERIA_NOT_LIST = "criteria_not_list"
    CRITERION_NOT_DICT = "criterion_not_dict"
    CRITERION_MISSING_FIELDS = "criterion_missing_fields"
    CRITERION_MISSING_SCORE_LEVEL = "criterion_missing_score_level"

    # creative_writing_caller.py post-parse checks (training reward path)
    WRITING_PROMPT_FIELDS = "writing_prompt_fields"
    EMPTY_QUERY_OR_CRITERIA = "empty_query_or_criteria"
