"""Tests for batch_eval_agent.py::PerCriterionEvalAgent — the judge.type=
sft-critic scoring strategy (REFACTOR_PLAN.md §6.3/§6.4).

Exercised against a fake agent implementing ClaudeAgent/MockJudgeAgent's
`.run()` surface — not against CriticServerAgent or real vLLM — so these
tests prove PerCriterionEvalAgent is generic across judge backends, per its
own docstring, and run with no network/GPU."""

import json
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
    # batch_eval_agent.py -> evaluator.llm imports requests/dotenv only, but
    # evaluator/__init__.py (imported transitively by nothing here) is not
    # touched by this file; kept for parity with the other WB test files
    # in case that changes.
    if "vllm" not in sys.modules:
        fake = types.ModuleType("vllm")
        fake.LLM = object
        fake.SamplingParams = object
        monkeypatch.setitem(sys.modules, "vllm", fake)


class _FakeAgent:
    """Records every prompt it's asked to score and returns a scripted
    response per call, mirroring ClaudeAgent/MockJudgeAgent/CriticServerAgent's
    `.run(prompt, max_try, success_check_fn)` surface."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def run(self, prompt, max_try=3, success_check_fn=None, **kwargs):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        success = success_check_fn(response) if success_check_fn else True
        return response, success


def _score_json(score, reason="fine"):
    return json.dumps({"score": score, "reason": reason})


CRITERIA = [
    {"name": "Clarity", "criteria_description": "Is it clear?"},
    {"name": "Originality", "criteria_description": "Is it original?"},
]


def test_one_judge_call_per_criterion():
    from batch_eval_agent import PerCriterionEvalAgent

    fake = _FakeAgent([_score_json(7), _score_json(4)])
    agent = PerCriterionEvalAgent(fake)

    agent.score_all_criteria(
        content={"response": "a sample response"},
        query="write something",
        criteria=CRITERIA,
    )
    assert len(fake.prompts) == len(CRITERIA)


def test_request_embeds_raw_criterion_dict_not_formatted_block():
    """Fidelity requirement (REFACTOR_PLAN.md §6.4): the criterion dict goes
    into the prompt via str.format's default repr, exactly like upstream
    WritingBench's EvalAgent.generate_score — not BatchEvalAgent's
    _format_criterion "Name: X\\nDescription: Y" presentation."""
    from batch_eval_agent import PerCriterionEvalAgent

    fake = _FakeAgent([_score_json(7), _score_json(4)])
    agent = PerCriterionEvalAgent(fake)

    agent.score_all_criteria(
        content={"response": "a sample response"},
        query="write something",
        criteria=CRITERIA,
    )

    first_prompt = fake.prompts[0]
    assert str(CRITERIA[0]) in first_prompt
    assert "Name: Clarity" not in first_prompt  # not _format_criterion's block
    assert "write something" in first_prompt
    assert "a sample response" in first_prompt


def test_response_reconstructed_keyed_by_criterion_name():
    from batch_eval_agent import PerCriterionEvalAgent

    fake = _FakeAgent([_score_json(7, "clear enough"), _score_json(4, "derivative")])
    agent = PerCriterionEvalAgent(fake)

    scores = agent.score_all_criteria(
        content={"response": "a sample response"},
        query="write something",
        criteria=CRITERIA,
    )

    assert scores == {
        "Clarity": {"score": 7, "reason": "clear enough"},
        "Originality": {"score": 4, "reason": "derivative"},
    }


def test_output_shape_matches_batch_eval_agent_exactly():
    """The whole point of the shared contract (REFACTOR_PLAN.md §6.3): reward
    callers must not need to know which scorer produced the dict."""
    from batch_eval_agent import BatchEvalAgent, PerCriterionEvalAgent
    from evaluator.mock import MockJudgeAgent

    content = {"response": "a sample response"}
    query = "write something"

    batched = BatchEvalAgent(MockJudgeAgent(system_prompt="sys")).score_all_criteria(
        content=content, query=query, criteria=CRITERIA
    )
    per_criterion = PerCriterionEvalAgent(MockJudgeAgent(system_prompt="sys")).score_all_criteria(
        content=content, query=query, criteria=CRITERIA
    )

    assert set(batched) == set(per_criterion) == {"Clarity", "Originality"}
    for shape in (batched, per_criterion):
        for v in shape.values():
            assert isinstance(v["score"], int)
            assert isinstance(v["reason"], str)


def test_missing_required_field_raises_judge_input_error_before_any_call():
    from batch_eval_agent import JudgeInputError, PerCriterionEvalAgent

    fake = _FakeAgent([_score_json(7)])
    agent = PerCriterionEvalAgent(fake)

    with pytest.raises(JudgeInputError):
        agent.score_all_criteria(
            content={"response": "x"},
            query="q",
            criteria=[{"name": "Clarity"}],  # missing criteria_description
        )
    assert fake.prompts == []  # no judge call made


def test_criterion_parse_failure_raises_judge_parse_error():
    from batch_eval_agent import JudgeParseError, PerCriterionEvalAgent

    fake = _FakeAgent(["not valid json"])
    agent = PerCriterionEvalAgent(fake)

    with pytest.raises(JudgeParseError):
        agent.score_all_criteria(
            content={"response": "x"},
            query="q",
            criteria=[CRITERIA[0]],
            max_retries=1,
        )


def test_criterion_api_error_propagates_immediately_no_partial_result():
    from batch_eval_agent import PerCriterionEvalAgent

    class _RaisingAgent:
        def run(self, prompt, max_try=3, success_check_fn=None, **kwargs):
            from evaluator.llm import JudgeAPIError
            raise JudgeAPIError("boom")

    agent = PerCriterionEvalAgent(_RaisingAgent())
    from evaluator.llm import JudgeAPIError

    with pytest.raises(JudgeAPIError):
        agent.score_all_criteria(
            content={"response": "x"},
            query="q",
            criteria=CRITERIA,
        )


def test_wraps_critic_server_agent_shaped_object_without_importing_it():
    """PerCriterionEvalAgent must be backend-agnostic — verified here by
    wrapping a plain fake with no relation to CriticServerAgent at all."""
    from batch_eval_agent import PerCriterionEvalAgent

    fake = _FakeAgent([_score_json(9)])
    scores = PerCriterionEvalAgent(fake).score_all_criteria(
        content={"response": "x"}, query="q", criteria=[CRITERIA[0]]
    )
    assert scores == {"Clarity": {"score": 9, "reason": "fine"}}
