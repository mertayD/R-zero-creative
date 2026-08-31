"""Tests for the challenger eval harness's judge/diversity/aggregate logic —
no GPU, no network. Judge scoring goes through MockJudgeAgent, the same
offline stand-in test_mock_judge.py uses for the WritingBench judge path."""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WB_DIR = REPO_ROOT / "evaluation" / "writing_bench"


@pytest.fixture(autouse=True)
def wb_on_path(monkeypatch):
    for p in (str(REPO_ROOT), str(WB_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # evaluator/__init__.py also imports CriticAgent (evaluator/critic.py),
    # which imports vllm — not installed/needed here, same workaround
    # test_mock_judge.py uses for the reward-caller imports.
    if "vllm" not in sys.modules:
        fake = types.ModuleType("vllm")
        fake.LLM = object
        fake.SamplingParams = object
        monkeypatch.setitem(sys.modules, "vllm", fake)


def _criteria():
    levels = {lvl: "..." for lvl in ("1-2", "3-4", "5-6", "7-8", "9-10")}
    return [{"name": "clarity", "criteria_description": "how clear it is", **levels}]


class _FakeLabelAgent:
    """Returns a fixed canned response regardless of prompt — used to test
    challenger_judge_agent's label -> numeric conversion without a real
    judge call. Matches ClaudeAgent/MockJudgeAgent's `.run(...) -> (response,
    success)` surface."""

    def __init__(self, response: str):
        self._response = response

    def run(self, prompt, temperature=1.0, max_length=None, max_try=5, success_check_fn=None):
        success = success_check_fn(self._response) if success_check_fn else True
        return self._response, success


# =============================================================================
# challenger_judge_agent
# =============================================================================

def test_score_generated_prompt_via_mock_agent_returns_all_three_dimensions():
    from creative_rzero.eval.challenger import challenger_judge_agent

    agent = challenger_judge_agent.get_agent("mock")
    result = challenger_judge_agent.score_generated_prompt(
        agent,
        domain_name="Academic & Engineering",
        subdomain="short story",
        guidance_applied=["Add a requirement for generating specific lengths."],
        query="write a short story about a lighthouse keeper",
        criteria=_criteria(),
    )

    assert result["judge_backend"] == "mock"
    for key in ("domain_adherence", "guidance_adherence", "criteria_quality"):
        assert 1 <= result[key] <= 5


def test_get_agent_rejects_unknown_judge_type():
    from creative_rzero.eval.challenger import challenger_judge_agent

    with pytest.raises(ValueError, match="sft-critic"):
        challenger_judge_agent.get_agent("sft-critic")


def test_get_agent_claude_requires_api_key(monkeypatch):
    from creative_rzero.eval.challenger import challenger_judge_agent

    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PERPLEXITY_API_KEY"):
        challenger_judge_agent.get_agent("claude")


def test_get_agent_claude_constructs_with_api_key(monkeypatch):
    from creative_rzero.eval.challenger import challenger_judge_agent

    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    agent = challenger_judge_agent.get_agent("claude")

    assert agent.system_prompt == challenger_judge_agent.judge_prompts.SYSTEM_PROMPT


def test_score_generated_prompt_converts_labels_to_numeric_values():
    from creative_rzero.eval.challenger import challenger_judge_agent

    response = (
        "```json\n"
        '{"domain_adherence": "MOSTLY_SATISFIED", "guidance_adherence": "NOT_SATISFIED", '
        '"criteria_quality": "PERFECTLY_SATISFIED", "reasoning": "because"}\n'
        "```"
    )

    result = challenger_judge_agent.score_generated_prompt(
        _FakeLabelAgent(response),
        domain_name="Academic & Engineering",
        subdomain="short story",
        guidance_applied=[],
        query="write a short story",
        criteria=_criteria(),
    )

    assert result == {
        "domain_adherence": 4,
        "guidance_adherence": 1,
        "criteria_quality": 5,
        "reasoning": "because",
        "judge_backend": "real",
    }


def test_score_generated_prompt_raises_on_unrecognized_label():
    from creative_rzero.eval.challenger import challenger_judge_agent

    bad_response = (
        '```json\n{"domain_adherence": "GREAT", "guidance_adherence": "GREAT", '
        '"criteria_quality": "GREAT"}\n```'
    )

    with pytest.raises(challenger_judge_agent.JudgeParseError):
        challenger_judge_agent.score_generated_prompt(
            _FakeLabelAgent(bad_response),
            domain_name="Academic & Engineering",
            subdomain="short story",
            guidance_applied=[],
            query="write a short story",
            criteria=_criteria(),
            max_retries=1,
        )


# =============================================================================
# diversity
# =============================================================================

def test_near_duplicate_pairs_flags_near_identical_queries():
    from creative_rzero.eval.challenger.diversity import near_duplicate_pairs

    queries = [
        "Write a formal quarterly report for the finance team about Q3 revenue.",
        "Write a formal quarterly report for the finance team about Q3 revenue!",
        "Compose a whimsical bedtime story about a dragon who is afraid of the dark.",
    ]
    pairs = near_duplicate_pairs(queries)

    assert pairs
    assert pairs[0][:2] == (0, 1)


def test_near_duplicate_pairs_empty_for_diverse_queries():
    from creative_rzero.eval.challenger.diversity import near_duplicate_pairs

    queries = [
        "Compose a whimsical bedtime story about a dragon who is afraid of the dark.",
        "Draft a technical spec for a distributed rate limiter used in a payments API.",
        "Write a persuasive op-ed arguing for stricter urban zoning reform.",
    ]
    assert near_duplicate_pairs(queries) == []


def test_duplicate_rate_handles_fewer_than_two_queries():
    from creative_rzero.eval.challenger.diversity import duplicate_rate

    assert duplicate_rate([]) == 0.0
    assert duplicate_rate(["only one query"]) == 0.0


def test_duplicate_rate_matches_flagged_fraction():
    from creative_rzero.eval.challenger.diversity import duplicate_rate

    queries = [
        "Write a formal quarterly report for the finance team about Q3 revenue.",
        "Write a formal quarterly report for the finance team about Q3 revenue!",
        "Compose a whimsical bedtime story about a dragon who is afraid of the dark.",
        "Draft a technical spec for a distributed rate limiter used in a payments API.",
    ]
    assert duplicate_rate(queries) == pytest.approx(0.5)


def test_add_diversity_records_partner_and_similarity():
    from creative_rzero.eval.challenger.run_eval import add_diversity

    def _r(eval_id, query, valid=True):
        return {
            "eval_id": eval_id, "domain": "D1", "subdomain": "short story",
            "format_valid": valid, "query": query,
        }

    scored = [
        _r("D1|short story|0", "Write a formal quarterly report for the finance team about Q3 revenue."),
        _r("D1|short story|1", "Write a formal quarterly report for the finance team about Q3 revenue!"),
        _r("D1|short story|2", "Compose a whimsical bedtime story about a dragon afraid of the dark."),
        _r("D1|short story|3", "", valid=False),
    ]
    add_diversity(scored, method="tfidf")

    assert scored[0]["near_duplicate"] and scored[1]["near_duplicate"]
    assert scored[0]["near_duplicate_of"] == "D1|short story|1"
    assert scored[1]["near_duplicate_of"] == "D1|short story|0"
    assert scored[0]["near_duplicate_similarity"] >= 0.32
    assert scored[0]["near_duplicate_similarity"] == scored[1]["near_duplicate_similarity"]

    assert scored[2]["near_duplicate"] is False
    assert scored[2]["near_duplicate_of"] is None and scored[2]["near_duplicate_similarity"] is None

    assert scored[3]["near_duplicate"] is None
    assert scored[3]["near_duplicate_of"] is None and scored[3]["near_duplicate_similarity"] is None


def test_add_diversity_embedding_method_uses_semantic_pairs(monkeypatch):
    from creative_rzero.eval.challenger import run_eval

    def fake_semantic_pairs(queries, threshold=None):
        return [(0, 1, 0.91)]

    monkeypatch.setattr(run_eval, "semantic_near_duplicate_pairs", fake_semantic_pairs)
    scored = [
        {"eval_id": "D1|poetry|0", "domain": "D1", "subdomain": "poetry",
         "format_valid": True, "query": "a"},
        {"eval_id": "D1|poetry|1", "domain": "D1", "subdomain": "poetry",
         "format_valid": True, "query": "b"},
        {"eval_id": "D1|poetry|2", "domain": "D1", "subdomain": "poetry",
         "format_valid": True, "query": "c"},
    ]
    run_eval.add_diversity(scored)  # embedding is the default

    assert scored[0]["near_duplicate"] and scored[1]["near_duplicate"]
    assert scored[0]["near_duplicate_of"] == "D1|poetry|1"
    assert scored[0]["near_duplicate_similarity"] == 0.91
    assert scored[2]["near_duplicate"] is False


def test_add_diversity_rejects_unknown_method():
    from creative_rzero.eval.challenger.run_eval import add_diversity

    with pytest.raises(ValueError, match="method"):
        add_diversity([], method="levenshtein")


# =============================================================================
# aggregate
# =============================================================================

def _row(**overrides):
    base = {
        "domain": "D1",
        "subdomain": "short story",
        "format_valid": True,
        "format_failure_reason": "ok",
        "domain_adherence": 4,
        "guidance_adherence": 3,
        "criteria_quality": 5,
        "query_len": 120,
        "criteria_len": 800,
        "near_duplicate": False,
    }
    base.update(overrides)
    return base


def test_aggregate_rows_computes_overall_and_stratified_stats():
    from creative_rzero.eval.challenger.aggregate import aggregate_rows

    rows = [
        _row(),
        _row(subdomain="poetry", domain_adherence=2, near_duplicate=True),
        _row(
            domain="D2",
            subdomain="business plan",
            format_valid=False,
            format_failure_reason="missing_json_fence",
            domain_adherence=None,
            guidance_adherence=None,
            criteria_quality=None,
            near_duplicate=None,
        ),
    ]

    summary = aggregate_rows(rows)

    assert summary["overall"]["n"] == 3
    assert summary["overall"]["format_pass_rate"] == pytest.approx(2 / 3)
    assert summary["overall"]["format_failure_reason_counts"] == {"missing_json_fence": 1}
    assert summary["overall"]["domain_adherence_mean"] == pytest.approx(3.0)
    assert summary["overall"]["duplicate_rate"] == pytest.approx(0.5)

    assert summary["by_domain"]["D1"]["n"] == 2
    assert summary["by_domain"]["D2"]["format_pass_rate"] == 0.0
    assert summary["by_domain"]["D2"]["domain_adherence_mean"] is None

    assert "D1::short story" in summary["by_subdomain"]
    assert "D1::poetry" in summary["by_subdomain"]
    assert summary["by_subdomain"]["D1::short story"]["n"] == 1


def test_aggregate_rows_empty_group_has_none_stats_not_a_crash():
    from creative_rzero.eval.challenger.aggregate import aggregate_rows

    summary = aggregate_rows([])

    assert summary["overall"]["n"] == 0
    assert summary["overall"]["format_pass_rate"] is None
    assert summary["overall"]["domain_adherence_mean"] is None
