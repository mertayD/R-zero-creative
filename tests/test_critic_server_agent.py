"""Tests for evaluator/critic_server.py::CriticServerAgent — the HTTP
client for judge.type=sft-critic. No real vLLM/Modal involved: requests.post
is monkeypatched with fake responses, matching the mock-first testing
philosophy already established for ClaudeAgent-shaped judges
(REFACTOR_PLAN.md §5/T3.14)."""

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
    # test_mock_judge.py/test_verl_entry.py use.
    if "vllm" not in sys.modules:
        fake = types.ModuleType("vllm")
        fake.LLM = object
        fake.SamplingParams = object
        monkeypatch.setitem(sys.modules, "vllm", fake)


@pytest.fixture(autouse=True)
def critic_env(monkeypatch):
    monkeypatch.setenv("WB_CRITIC_URL", "https://fake-critic.modal.run")
    monkeypatch.delenv("CRITIC_JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("JUDGE_MAX_HTTP_RETRY_ATTEMPTS", "3")
    # _MAX_HTTP_RETRY_ATTEMPTS is read once at module import time (mirrors
    # evaluator/llm.py's own constant) — if some earlier test already
    # imported evaluator.critic_server (e.g. transitively via
    # evaluator/__init__.py) before this env var was set, reload so this
    # test's value actually takes effect.
    if "evaluator.critic_server" in sys.modules:
        importlib.reload(sys.modules["evaluator.critic_server"])


class _FakeResponse:
    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.text = text

    def json(self):
        return self._json_body


def _chat_completion_body(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_requires_critic_url(monkeypatch):
    monkeypatch.delenv("WB_CRITIC_URL", raising=False)
    from evaluator.critic_server import CriticServerAgent

    with pytest.raises(RuntimeError, match="WB_CRITIC_URL"):
        CriticServerAgent(system_prompt="test")


def test_success_path_returns_response_and_true(monkeypatch):
    from evaluator.critic_server import CriticServerAgent

    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append((url, json, headers))
        return _FakeResponse(200, _chat_completion_body('{"score": 7, "reason": "ok"}'))

    monkeypatch.setattr("evaluator.critic_server.requests.post", fake_post)

    agent = CriticServerAgent(system_prompt="sys")
    response, success = agent.run(prompt="score this", success_check_fn=lambda r: True)

    assert success is True
    assert response == '{"score": 7, "reason": "ok"}'
    assert len(calls) == 1
    url, payload, headers = calls[0]
    assert url == "https://fake-critic.modal.run/v1/chat/completions"
    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "score this"},
    ]
    assert headers == {}  # no CRITIC_JUDGE_API_KEY set


def test_bearer_token_sent_when_api_key_configured(monkeypatch):
    monkeypatch.setenv("CRITIC_JUDGE_API_KEY", "secret-token")
    from evaluator.critic_server import CriticServerAgent

    seen_headers = {}

    def fake_post(url, json, headers, timeout):
        seen_headers.update(headers)
        return _FakeResponse(200, _chat_completion_body("x"))

    monkeypatch.setattr("evaluator.critic_server.requests.post", fake_post)

    CriticServerAgent(system_prompt="sys").run(prompt="p")
    assert seen_headers == {"Authorization": "Bearer secret-token"}


def test_429_triggers_backoff_not_immediate_failure(monkeypatch):
    from evaluator.critic_server import CriticServerAgent

    monkeypatch.setattr("evaluator.critic_server.time.sleep", lambda _: None)
    responses = [
        _FakeResponse(429, text="rate limited"),
        _FakeResponse(200, _chat_completion_body("ok after retry")),
    ]

    def fake_post(url, json, headers, timeout):
        return responses.pop(0)

    monkeypatch.setattr("evaluator.critic_server.requests.post", fake_post)

    agent = CriticServerAgent(system_prompt="sys")
    response, success = agent.run(prompt="p")
    assert success is True
    assert response == "ok after retry"


def test_retry_exhaustion_raises_judge_api_error_with_last_status(monkeypatch):
    from evaluator.critic_server import CriticServerAgent
    from evaluator.llm import JudgeAPIError

    monkeypatch.setattr("evaluator.critic_server.time.sleep", lambda _: None)

    def fake_post(url, json, headers, timeout):
        return _FakeResponse(503, text="model loading")

    monkeypatch.setattr("evaluator.critic_server.requests.post", fake_post)

    agent = CriticServerAgent(system_prompt="sys")
    with pytest.raises(JudgeAPIError) as exc_info:
        agent.run(prompt="p")
    assert exc_info.value.status_code == 503


def test_network_error_retries_then_raises(monkeypatch):
    import requests

    from evaluator.critic_server import CriticServerAgent
    from evaluator.llm import JudgeAPIError

    monkeypatch.setattr("evaluator.critic_server.time.sleep", lambda _: None)

    def fake_post(url, json, headers, timeout):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr("evaluator.critic_server.requests.post", fake_post)

    agent = CriticServerAgent(system_prompt="sys")
    with pytest.raises(JudgeAPIError):
        agent.run(prompt="p")


def test_success_check_fn_retried_before_success(monkeypatch):
    """run()'s own retry loop (separate from the HTTP-level retry inside
    call_critic) — a response that fails success_check_fn is retried up to
    max_try, matching ClaudeAgent.run's contract."""
    from evaluator.critic_server import CriticServerAgent

    bodies = ["not json", '{"score": 5, "reason": "ok"}']

    def fake_post(url, json, headers, timeout):
        return _FakeResponse(200, _chat_completion_body(bodies.pop(0)))

    monkeypatch.setattr("evaluator.critic_server.requests.post", fake_post)

    agent = CriticServerAgent(system_prompt="sys")
    response, success = agent.run(
        prompt="p",
        max_try=3,
        success_check_fn=lambda r: r.startswith("{"),
    )
    assert success is True
    assert response == '{"score": 5, "reason": "ok"}'
