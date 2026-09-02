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

    with pytest.raises(ValueError, match="gpt-4"):
        challenger_judge_agent.get_agent("gpt-4")


def test_get_agent_sft_critic_requires_server_url(monkeypatch):
    from creative_rzero.eval.challenger import challenger_judge_agent

    monkeypatch.delenv("WB_CRITIC_URL", raising=False)
    with pytest.raises(RuntimeError, match="WB_CRITIC_URL"):
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


def test_self_bleu_none_for_fewer_than_two_queries():
    from creative_rzero.eval.challenger.diversity import self_bleu

    assert self_bleu([]) is None
    assert self_bleu(["only one query"]) is None


def test_self_bleu_higher_for_near_identical_queries():
    from creative_rzero.eval.challenger.diversity import self_bleu

    near_identical = [
        "Write a formal quarterly report for the finance team about Q3 revenue.",
        "Write a formal quarterly report for the finance team about Q3 revenue!",
    ]
    diverse = [
        "Compose a whimsical bedtime story about a dragon who is afraid of the dark.",
        "Draft a technical spec for a distributed rate limiter used in a payments API.",
    ]

    assert self_bleu(near_identical) > self_bleu(diverse)


def test_pairwise_bleu_scores_returns_ordered_pairs():
    from creative_rzero.eval.challenger.diversity import pairwise_bleu_scores

    assert pairwise_bleu_scores([]) == []
    assert pairwise_bleu_scores(["only one query"]) == []

    queries = ["a b c", "a b d", "x y z"]
    scores = pairwise_bleu_scores(queries)
    assert len(scores) == 6  # n * (n - 1) ordered pairs


def test_all_pairwise_similarities_unfiltered_by_threshold():
    from creative_rzero.eval.challenger.diversity import all_pairwise_similarities, near_duplicate_pairs

    queries = [
        "Compose a whimsical bedtime story about a dragon who is afraid of the dark.",
        "Draft a technical spec for a distributed rate limiter used in a payments API.",
        "Write a persuasive op-ed arguing for stricter urban zoning reform.",
    ]
    # These are diverse enough that near_duplicate_pairs (thresholded) finds nothing...
    assert near_duplicate_pairs(queries) == []
    # ...but all_pairwise_similarities still returns every unordered pair.
    sims = all_pairwise_similarities(queries, method="tfidf")
    assert len(sims) == 3


def test_all_pairwise_similarities_rejects_unknown_method():
    from creative_rzero.eval.challenger.diversity import all_pairwise_similarities

    with pytest.raises(ValueError, match="method"):
        all_pairwise_similarities(["a", "b"], method="levenshtein")


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


def test_add_diversity_records_partners_and_count():
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
    group_histograms = add_diversity(scored, method="tfidf")

    assert scored[0]["near_duplicate"] and scored[1]["near_duplicate"]
    assert scored[0]["near_duplicate_partners"] == ["D1|short story|1"]
    assert scored[1]["near_duplicate_partners"] == ["D1|short story|0"]
    assert scored[0]["near_duplicate_count"] == 1
    assert scored[0]["near_duplicate_similarity"] >= 0.32
    assert scored[0]["near_duplicate_similarity"] == scored[1]["near_duplicate_similarity"]

    # group_self_bleu is a group-level stat: same value on every valid row in the group.
    assert scored[0]["group_self_bleu"] == scored[1]["group_self_bleu"] == scored[2]["group_self_bleu"]
    assert scored[0]["group_self_bleu"] > 0

    assert scored[2]["near_duplicate"] is False
    assert scored[2]["near_duplicate_partners"] == []
    assert scored[2]["near_duplicate_count"] == 0
    assert scored[2]["near_duplicate_similarity"] is None

    assert scored[3]["near_duplicate"] is None
    assert scored[3]["near_duplicate_partners"] is None
    assert scored[3]["near_duplicate_count"] is None
    assert scored[3]["near_duplicate_similarity"] is None
    assert scored[3]["group_self_bleu"] is None

    # 3 valid rows -> 3 unordered cosine pairs, 6 ordered BLEU pairs, unfiltered
    # by any threshold (unlike near_duplicate_partners, which only lists flagged ones).
    key = ("D1", "short story")
    assert set(group_histograms.keys()) == {key}
    assert len(group_histograms[key]["cosine_similarities"]) == 3
    assert len(group_histograms[key]["bleu_scores"]) == 6


def test_add_diversity_reports_all_partners_not_just_closest():
    from creative_rzero.eval.challenger.run_eval import add_diversity

    def _r(eval_id, query):
        return {
            "eval_id": eval_id, "domain": "D1", "subdomain": "short story",
            "format_valid": True, "query": query,
        }

    # Three near-identical queries: row 0 should be flagged against both 1 and 2,
    # not just whichever one happens to be closest.
    scored = [
        _r("D1|short story|0", "Write a formal quarterly report for the finance team about Q3 revenue."),
        _r("D1|short story|1", "Write a formal quarterly report for the finance team about Q3 revenue!"),
        _r("D1|short story|2", "Write a formal quarterly report for the finance team on Q3 revenue."),
    ]
    add_diversity(scored, method="tfidf")

    assert scored[0]["near_duplicate_count"] == 2
    assert set(scored[0]["near_duplicate_partners"]) == {"D1|short story|1", "D1|short story|2"}


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
    assert scored[0]["near_duplicate_partners"] == ["D1|poetry|1"]
    assert scored[0]["near_duplicate_count"] == 1
    assert scored[0]["near_duplicate_similarity"] == 0.91
    assert scored[2]["near_duplicate"] is False


def test_add_diversity_rejects_unknown_method():
    from creative_rzero.eval.challenger.run_eval import add_diversity

    with pytest.raises(ValueError, match="method"):
        add_diversity([], method="levenshtein")


# =============================================================================
# wandb similarity-histogram helpers (needs matplotlib + wandb; skipped otherwise)
# =============================================================================

pytest.importorskip("wandb")
pytest.importorskip("matplotlib")


def test_histogram_image_handles_empty_and_nonempty_values():
    from creative_rzero.eval.challenger.run_eval import _histogram_image

    assert _histogram_image([], "empty group").__class__.__name__ == "Image"
    assert _histogram_image([0.1, 0.5, 0.9], "nonempty group").__class__.__name__ == "Image"


def test_similarity_histogram_table_has_one_row_per_group():
    from creative_rzero.eval.challenger.run_eval import _similarity_histogram_table

    group_histograms = {
        ("D1", "short story"): {"cosine_similarities": [0.1, 0.9], "bleu_scores": [0.2, 0.3, 0.4]},
        ("D1", "poetry"): {"cosine_similarities": [], "bleu_scores": []},
    }
    table = _similarity_histogram_table(group_histograms)

    assert len(table.data) == 2
    # sorted() by (domain, subdomain): "poetry" < "short story"
    assert table.data[0][:3] == ["D1", "poetry", 0]
    assert table.data[1][:3] == ["D1", "short story", 2]
    assert table.data[1][4] == 3  # n_bleu_pairs


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
        "group_self_bleu": 0.4,
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
            group_self_bleu=None,
        ),
    ]

    summary = aggregate_rows(rows)

    assert summary["overall"]["n"] == 3
    assert summary["overall"]["format_pass_rate"] == pytest.approx(2 / 3)
    assert summary["overall"]["format_failure_reason_counts"] == {"missing_json_fence": 1}
    assert summary["overall"]["domain_adherence_mean"] == pytest.approx(3.0)
    assert summary["overall"]["duplicate_rate"] == pytest.approx(0.5)
    assert summary["overall"]["self_bleu_mean"] == pytest.approx(0.4)

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
    assert summary["overall"]["self_bleu_mean"] is None
