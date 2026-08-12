"""Fidelity guard for evaluation/writing_bench/per_criterion_eval_prompt.py —
must stay byte-identical to the upstream single-criterion prompt
(X-PLUG/WritingBench/blob/main/prompt.py), which the sft-critic judge
(judge.type=sft-critic) was trained/validated against. See
docs/REFACTOR_PLAN.md §6.4.

Pinned to a known-good copy fetched during design rather than fetched live —
tests must not require network access, per this repo's mock-first testing
philosophy (REFACTOR_PLAN.md §5)."""

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WB_DIR = REPO_ROOT / "evaluation" / "writing_bench"

if str(WB_DIR) not in sys.path:
    sys.path.insert(0, str(WB_DIR))

# sha256 of evaluate_system + "\n" + evaluate_prompt, computed against the
# upstream content fetched from
# https://raw.githubusercontent.com/X-PLUG/WritingBench/main/prompt.py
# during design (2026-08-11). Also matches the already-vendored
# evaluation/writing_bench/prompt.py used by evaluate_benchmark.py.
_EXPECTED_SHA256 = "716d40e5cdfb367b9ba2fa06a8cd8638cd31448281062c24ad346214d2ca824f"


def _content_hash() -> str:
    from per_criterion_eval_prompt import evaluate_prompt, evaluate_system

    payload = (evaluate_system + "\n" + evaluate_prompt).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_matches_already_vendored_prompt_py():
    """per_criterion_eval_prompt.py and prompt.py are independent, deliberate
    copies of the same upstream file (see per_criterion_eval_prompt.py's
    module docstring) — they must still agree exactly."""
    import per_criterion_eval_prompt as per_criterion
    import prompt as standalone_bench_prompt

    assert per_criterion.evaluate_system == standalone_bench_prompt.evaluate_system
    assert per_criterion.evaluate_prompt == standalone_bench_prompt.evaluate_prompt


def test_prompt_content_is_pinned():
    """Guards against an accidental future edit to the vendored file — if
    this ever fails, either the edit was unintentional (revert it) or it was
    a deliberate re-vendor from upstream (update _EXPECTED_SHA256 here)."""
    assert _content_hash() == _EXPECTED_SHA256


def test_prompt_has_expected_placeholders():
    from per_criterion_eval_prompt import evaluate_prompt

    for placeholder in ("{criteria}", "{query}", "{response}"):
        assert placeholder in evaluate_prompt


def test_prompt_output_format_requests_single_score_and_reason():
    """Not the batched (multi-criterion-in-one-JSON) schema
    batch_eval_prompt.py uses — a single {"score", "reason"} object."""
    from per_criterion_eval_prompt import evaluate_prompt

    assert '"score":' in evaluate_prompt
    assert '"reason":' in evaluate_prompt
