# Vendored verbatim from https://github.com/X-PLUG/WritingBench (Apache-2.0).
# Provides the official CriticAgent (AQuarterMile/WritingBench-Critic-Model-Qwen-7B
# served via vLLM). Not used by default in this project (we use Claude-Sonnet-4-5
# per the current leaderboard), but kept available behind --evaluator critic so the
# self-hostable judge path remains a one-flag switch.

import time
from typing import Callable
from vllm import LLM, SamplingParams


class CriticAgent(object):
    def __init__(self,
                 system_prompt: str = None):
        self.system_prompt = system_prompt
        self.model = LLM(
            model="",  # Local path to AQuarterMile/WritingBench-Critic-Model-Qwen-7B
            tensor_parallel_size=1,
        )

    def call_critic(self,
                    messages: str,
                    top_p: float = 0.95,
                    temperature: float = 1.0,
                    max_length: int = 2048):

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=int(max_length),
        )

        attempt = 0
        max_attempts = 5
        wait_time = 1

        while attempt < max_attempts:
            try:
                response = self.model.chat(messages, sampling_params)
                return response[0].outputs[0].text
            except Exception as e:
                print(f"Attempt {attempt+1}: VLLM call failed due to error: {e}, retrying...")

            time.sleep(wait_time)
            attempt += 1

        raise Exception("Max attempts exceeded. Failed to get a successful response.")

    def basic_success_check(self, response):
        if not response:
            print(response)
            return False
        return True

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

        while try_times < max_try:
            response = self.call_critic(
                messages=messages,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
            )
            if success_check_fn is None:
                success_check_fn = lambda x: True
            if success_check_fn(response):
                success = True
                break
            else:
                try_times += 1

        return response, success
