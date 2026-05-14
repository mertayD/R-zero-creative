"""
ClaudeAgent — official WritingBench LLM-as-judge path.

Adapted from upstream evaluator/llm.py (Apache-2.0). Upstream's stub left api_key,
url, and model blank for the user to fill in against an OpenAI-compatible
endpoint. We rewrite the body to call Anthropic's Messages API directly with
Claude-Sonnet-4-5, which is the judge the WritingBench leaderboard switched to
on 2025-11-27. Public scores on the leaderboard are produced this way, so this
keeps our results directly comparable.

Sampling parameters are kept exactly as upstream (top_p=0.95, temperature=1.0,
max_length=2048) so scores are comparable.

Configuration:
    ANTHROPIC_API_KEY    required, your Anthropic key
    WB_JUDGE_MODEL       optional, default 'claude-sonnet-4-5'
"""

import os
import time
from typing import Callable

import requests


# Default model id matches the one currently used by the WritingBench leaderboard.
DEFAULT_JUDGE_MODEL = os.environ.get("WB_JUDGE_MODEL", "claude-sonnet-4-5")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"


class ClaudeAgent(object):
    """
    Drop-in replacement for the upstream ClaudeAgent. The class name is kept so
    evaluator/__init__.py and evaluate_benchmark.py can import it unchanged.
    """

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = DEFAULT_JUDGE_MODEL
        self.url = ANTHROPIC_API_URL
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running the "
                "WritingBench Claude judge: `export ANTHROPIC_API_KEY=...`"
            )

    def call_claude(self,
                    messages,
                    top_p: float = 0.95,
                    temperature: float = 1.0,
                    max_length: int = 2048):
        # Anthropic's Messages API takes `system` as a top-level field, not a
        # role inside `messages`. Strip a leading system message if present and
        # promote it to the top-level field.
        system = self.system_prompt
        msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                msgs.append({"role": m["role"], "content": m["content"]})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        # Anthropic's API rejects requests that specify both `temperature` and
        # `top_p`. Upstream WritingBench's evaluator/llm.py sends both because
        # it was written against an OpenAI-compatible endpoint where that's
        # allowed; against Anthropic's native /v1/messages endpoint we have to
        # pick one. WritingBench's spec is top_p=0.95, temperature=1.0. Since
        # temperature=1.0 is also Anthropic's default, sending it is a no-op —
        # the actually-meaningful parameter is top_p=0.95, so we keep that.
        data = {
            "model": self.model,
            "max_tokens": int(max_length),
            "top_p": top_p,
            "messages": msgs,
        }
        if system:
            data["system"] = system

        attempt = 0
        max_attempts = 5
        wait_time = 1

        while attempt < max_attempts:
            try:
                response = requests.post(self.url, headers=headers, json=data, timeout=120)
                if response.status_code == 200:
                    body = response.json()
                    # Anthropic returns: {"content":[{"type":"text","text":"..."}, ...], ...}
                    parts = body.get("content", [])
                    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
                    return text
                else:
                    print(f"Attempt {attempt+1}: HTTP {response.status_code}: {response.text[:300]}")
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}: network error: {e}")

            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 30)
            attempt += 1

        raise Exception("Max attempts exceeded. Failed to get a successful response from Anthropic.")

    def basic_success_check(self, response):
        return bool(response)

    def run(self,
            prompt: str,
            top_p: float = 0.95,
            temperature: float = 1.0,
            max_length: int = 2048,
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
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
            )
            check = success_check_fn or (lambda x: True)
            if check(response):
                success = True
                break
            try_times += 1

        return response, success
