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
