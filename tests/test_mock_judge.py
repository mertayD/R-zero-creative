"""Tests for MockJudgeAgent (evaluation/writing_bench/evaluator/mock.py),
the judge.type=mock offline stand-in for ClaudeAgent.

Exercised through the real BatchEvalAgent — not a mock of BatchEvalAgent
itself — to prove MockJudgeAgent is a genuine drop-in for ClaudeAgent that
needs no changes to batch_eval_agent.py."""

import importlib
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
    # test_verl_entry.py uses for the reward-caller imports.
    if "vllm" not in sys.modules:
        fake = types.ModuleType("vllm")
        fake.LLM = object
        fake.SamplingParams = object
        monkeypatch.setitem(sys.modules, "vllm", fake)


def test_mock_agent_scores_via_real_batch_eval_agent():
    from batch_eval_agent import BatchEvalAgent
    from evaluator.mock import MockJudgeAgent

    agent = BatchEvalAgent(MockJudgeAgent(system_prompt="test"))
    criteria = [
        {"name": "Clarity", "criteria_description": "Is it clear?"},
        {"name": "Originality", "criteria_description": "Is it original?"},
    ]
    scores = agent.score_all_criteria(
        content={"response": "a sample response"},
        query="write something",
        criteria=criteria,
    )
    assert set(scores) == {"Clarity", "Originality"}
    for v in scores.values():
        assert isinstance(v["score"], int)
        assert 1 <= v["score"] <= 10
        assert isinstance(v["reason"], str)


def test_mock_agent_reproducible_with_same_seed(monkeypatch):
    monkeypatch.setenv("WB_JUDGE_MOCK_SEED", "7")
    import evaluator.mock as mock_mod
    importlib.reload(mock_mod)

    prompt = "Name: Clarity\nName: Originality\n"
    r1, ok1 = mock_mod.MockJudgeAgent().run(prompt)
    r2, ok2 = mock_mod.MockJudgeAgent().run(prompt)
    assert ok1 and ok2
    assert r1 == r2


def test_mock_agent_ignores_name_lines_in_response_body():
    """A response containing a line that happens to start with "Name: "
    (e.g. dialogue) must not leak into the extracted criteria keys — only
    the "** Criteria **" section, not the whole prompt, is scanned."""
    from batch_eval_agent import BatchEvalAgent
    from evaluator.mock import MockJudgeAgent

    agent = BatchEvalAgent(MockJudgeAgent(system_prompt="test"))
    criteria = [{"name": "Clarity", "criteria_description": "Is it clear?"}]
    scores = agent.score_all_criteria(
        content={"response": 'She said, "Name: Mr. Smith," and walked away.'},
        query="write something",
        criteria=criteria,
    )
    assert set(scores) == {"Clarity"}


def test_mock_agent_requires_no_api_key(monkeypatch):
    """Unlike ClaudeAgent, construction must not require PERPLEXITY_API_KEY —
    that's the entire point of the mock path."""
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    from evaluator.mock import MockJudgeAgent

    MockJudgeAgent(system_prompt="test")  # must not raise
