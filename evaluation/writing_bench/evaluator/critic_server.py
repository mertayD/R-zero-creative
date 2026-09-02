"""
CriticServerAgent — HTTP client for a self-hosted vLLM sft-critic judge.

Selected via `judge.type: sft-critic` (creative_rzero/config.py VALID_JUDGE_TYPES).
Reaches a persistent Modal-deployed vLLM OpenAI-compatible server hosting
AQuarterMile/WritingBench-Critic-Model-Qwen-7B (see modal_app.py::critic_judge_server
and docs/REFACTOR_PLAN.md §6). Distinct from evaluator/critic.py::CriticAgent,
which loads vLLM in-process and is only used by the standalone benchmark
scripts (evaluate_benchmark.py / solver_sampling.py) — that file is untouched.

Matches ClaudeAgent's public surface (`run`) and failure semantics exactly
(same JudgeAPIError, same exponential-backoff-with-jitter retry shape) so
every failure_reason/backoff mechanism already built for the Claude judge
(REFACTOR_PLAN T1.4/T1.5) covers this backend for free.

No auth: the deployed server is only ever spun up/torn down around a
training run by us (min_containers=0 — REFACTOR_PLAN.md §6.2), not a
long-lived public service, so there's no bearer token here unlike
ClaudeAgent's required PERPLEXITY_API_KEY (a third-party API that mandates
one). If that deployment model ever changes, add auth then.

Configuration:
    WB_CRITIC_URL      required, base URL of the deployed vLLM sft-critic server
    WB_CRITIC_MODEL     optional, default 'writingbench-critic-qwen-7b' —
                         must match the server's --served-model-name
    JUDGE_MAX_HTTP_RETRY_ATTEMPTS  optional, default 5 (shared with ClaudeAgent)
"""

import os
import time
from typing import Callable

import requests

from .llm import JudgeAPIError

DEFAULT_CRITIC_MODEL = os.environ.get("WB_CRITIC_MODEL", "writingbench-critic-qwen-7b")

# Same env var ClaudeAgent reads — one shared HTTP retry budget knob for both
# backends, projected from judge.http_max_retry_attempts by
# verl_entry.py::_bridge_config_to_env.
_MAX_HTTP_RETRY_ATTEMPTS = int(os.environ.get("JUDGE_MAX_HTTP_RETRY_ATTEMPTS", "5"))


class CriticServerAgent(object):
    """Drop-in replacement for evaluator.llm.ClaudeAgent, backed by a
    self-hosted vLLM OpenAI-compatible chat completions endpoint instead of
    the Perplexity Gateway's Anthropic-Messages-compatible one."""

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt
        self.url = os.environ.get("WB_CRITIC_URL", "")
        self.model = DEFAULT_CRITIC_MODEL
        if not self.url:
            raise RuntimeError(
                "WB_CRITIC_URL is not set. judge.type=sft-critic requires "
                "judge.critic_url to be set in the experiment config."
            )

    def call_critic(
        self,
        messages,
        temperature: float = 1.0,
        max_length: int = 2048,
    ):
        endpoint = f"{self.url.rstrip('/')}/v1/chat/completions"
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": int(max_length),
        }

        attempt = 0
        wait_time = 1
        last_status_code = None

        while attempt < _MAX_HTTP_RETRY_ATTEMPTS:
            try:
                response = requests.post(endpoint, json=data, timeout=120)
                if response.status_code == 200:
                    body = response.json()
                    # OpenAI chat-completions schema:
                    # {"choices": [{"message": {"content": "..."}}], ...}
                    return body["choices"][0]["message"]["content"]
                else:
                    last_status_code = response.status_code
                    print(f"Attempt {attempt+1}: HTTP {response.status_code}: {response.text[:300]}")
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}: network error: {e}")

            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 30)
            attempt += 1

        raise JudgeAPIError(
            "Max attempts exceeded. Failed to get a successful response from the sft-critic server.",
            status_code=last_status_code,
        )

    def basic_success_check(self, response):
        return bool(response)

    def run(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_length: int = 2048,
        max_try: int = 5,
        success_check_fn: Callable = None,
    ):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        success = False
        try_times = 0
        response = None

        while try_times < max_try:
            response = self.call_critic(
                messages=messages,
                temperature=temperature,
                max_length=max_length,
            )
            check = success_check_fn or (lambda x: True)
            if check(response):
                success = True
                break
            try_times += 1

        return response, success
