"""
ClaudeAgent — official WritingBench LLM-as-judge path.

Adapted from upstream evaluator/llm.py (Apache-2.0). Upstream's stub left api_key,
url, and model blank for the user to fill in against an OpenAI-compatible
endpoint. We call Claude through Perplexity's **Agent API**
(`https://api.perplexity.ai/v1/agent`), which is the judge the WritingBench
leaderboard switched to on 2025-11-27. Public scores on the leaderboard are
produced this way, so this keeps our results directly comparable.

This is deliberately NOT Perplexity's Router API
(`https://api.perplexity.ai/router/v1/messages`), despite that one accepting
an Anthropic-Messages-shaped request/model id that looks like a closer match
— confirmed live (2026-08-19) that the Router API 403s with `"The Router API
is currently in limited preview. Contact api@perplexity.ai to request
access"` on this account, while the Agent API's `anthropic/<model>` routing
works today with no special access. The Agent API's request/response shape
is OpenAI-Responses-API-style, not Anthropic-Messages-style: `input` (not
`messages`), `instructions` (not a `system` message), `max_output_tokens`
(not `max_tokens`), and a response body shaped
`output: [{"type": "message", "content": [{"type": "output_text", "text":
...}]}]` (not a top-level `content` array). `call_claude` below still takes
the familiar `messages` list — kept for a stable interface across this
rewrite — and translates it into that shape internally.

Sampling parameters are kept close to upstream (temperature=1.0); max_length
is now a configurable default rather than a fixed 2048 — see WB_JUDGE_MAX_TOKENS.

Configuration:
    PERPLEXITY_API_KEY            required, your Perplexity Agent API key
    WB_JUDGE_MODEL                 optional, default 'claude-sonnet-4-5'
    WB_JUDGE_MAX_TOKENS            optional, default 8192
    JUDGE_MAX_HTTP_RETRY_ATTEMPTS  optional, default 5
"""

import os
import time
from typing import Callable

import requests
from dotenv import load_dotenv

load_dotenv()

# Default model id matches the one currently used by the WritingBench leaderboard.
DEFAULT_JUDGE_MODEL = os.environ.get("WB_JUDGE_MODEL", "claude-sonnet-5")
PERPLEXITY_AGENT_URL = "https://api.perplexity.ai/v1/agent"

# Judge sampling/response budget. This is a stopgap env-var knob, not the
# real config system — Phase 2 (creative_rzero/config.py) is where this and
# every other judge/reward knob become one reviewed, typed config. Until
# then it at least isn't a bare literal buried in a function signature.
DEFAULT_JUDGE_MAX_TOKENS = int(os.environ.get("WB_JUDGE_MAX_TOKENS", "8192"))

# Network/HTTP retry budget for a single call_claude invocation — separate
# from BatchEvalAgent's parse/validation retry budget (max_try in run(),
# one level up). Backoff is exponential (capped at 30s) with jitter. Was a
# bare literal (5) inline; now a named, env-overridable constant so it can
# be tuned without a code change.
_MAX_HTTP_RETRY_ATTEMPTS = int(os.environ.get("JUDGE_MAX_HTTP_RETRY_ATTEMPTS", "5"))


class JudgeAPIError(Exception):
    """Raised when the judge call fails at the network/HTTP layer (after
    call_claude's own retry budget is exhausted) — distinct from a
    JudgeParseError, which means the judge responded but its output never
    validated. Carries the last HTTP status code seen, if any, so callers
    can tell a 429 rate limit apart from a 5xx/network failure."""

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class ClaudeAgent(object):
    """
    Drop-in replacement for the upstream ClaudeAgent. The class name is kept so
    evaluator/__init__.py and evaluate_benchmark.py can import it unchanged.
    """

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt
        self.api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        self.model = f"anthropic/{DEFAULT_JUDGE_MODEL}"
        self.url = PERPLEXITY_AGENT_URL
        if not self.api_key:
            raise RuntimeError(
                "PERPLEXITY_API_KEY is not set. Export it before running the "
                "WritingBench Claude judge: `export PERPLEXITY_API_KEY=...`"
            )

    def call_claude(self,
                    messages,
                    temperature: float = 1.0,
                    max_length: int = DEFAULT_JUDGE_MAX_TOKENS):
        # The Agent API has no `system` role — a leading system message
        # becomes the top-level `instructions` field instead. Every caller in
        # this repo (ClaudeAgent.run, below) sends exactly one system + one
        # user message, so the common case is `input` = that one user
        # message's bare content string; the multi-message branch below only
        # exists so a future multi-turn caller degrades gracefully instead of
        # silently dropping messages.
        system = self.system_prompt
        rest = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                rest.append(m)

        if len(rest) == 1:
            input_text = rest[0]["content"]
        else:
            input_text = "\n\n".join(f"{m['role']}: {m['content']}" for m in rest)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        # temperature/top_p are deliberately not sent — temperature=1.0 is
        # already Anthropic's default via this route, so omitting it is a
        # no-op, and it keeps parity with the Router-API code path this
        # replaced (which omitted both for the same reason against a
        # different, Anthropic-Messages-shaped endpoint).
        data = {
            "model": self.model,
            "max_output_tokens": int(max_length),
            "input": input_text,
        }
        if system:
            data["instructions"] = system

        attempt = 0
        wait_time = 1
        last_status_code = None

        while attempt < _MAX_HTTP_RETRY_ATTEMPTS:
            try:
                response = requests.post(self.url, headers=headers, json=data, timeout=120)
                if response.status_code == 200:
                    body = response.json()
                    # Agent API returns an OpenAI-Responses-API-shaped body:
                    # {"output": [{"type": "message", "content": [{"type":
                    # "output_text", "text": "..."}, ...]}, ...], ...} — collect
                    # every output_text part across every message item, in
                    # order, the same "concatenate all text parts" contract
                    # the old Anthropic-Messages parsing had.
                    parts = []
                    for item in body.get("output", []):
                        if item.get("type") != "message":
                            continue
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                parts.append(c.get("text", ""))
                    return "".join(parts)
                else:
                    last_status_code = response.status_code
                    print(f"Attempt {attempt+1}: HTTP {response.status_code}: {response.text[:300]}")
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}: network error: {e}")

            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 30)
            attempt += 1

        raise JudgeAPIError(
            "Max attempts exceeded. Failed to get a successful response from the Perplexity Agent API.",
            status_code=last_status_code,
        )

    def basic_success_check(self, response):
        return bool(response)

    def run(self,
            prompt: str,
            temperature: float = 1.0,
            max_length: int = DEFAULT_JUDGE_MAX_TOKENS,
            max_try: int = 5,
            success_check_fn: Callable = None):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        success = False
        try_times = 0
        response = None

        while try_times < max_try:
            response = self.call_claude(
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
