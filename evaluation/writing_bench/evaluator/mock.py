"""
MockJudgeAgent — offline stand-in for ClaudeAgent (evaluator/llm.py).

Selected via `judge.type: mock` in the experiment config (creative_rzero/config.py
VALID_JUDGE_TYPES). Matches ClaudeAgent's public surface (`run`) exactly, so
BatchEvalAgent and both reward callers (creative_solver_caller.py,
creative_writing_caller.py) run unmodified — only the judge's network call is
replaced, not the scoring/ranking pipeline built on top of it. No
PERPLEXITY_API_KEY required.

Scores are pseudo-random per criterion (1-10, matching the real judge's
scale) but drawn from a seeded RNG so a run is reproducible: same config, same
scores, useful when debugging a pipeline issue without that noise also
varying between attempts. Override the seed with WB_JUDGE_MOCK_SEED.
"""

import json
import os
import random
import re
from typing import Callable

_NAME_RE = re.compile(r"^Name: (.+)$", re.MULTILINE)
_MOCK_SEED = int(os.environ.get("WB_JUDGE_MOCK_SEED", "42"))


class MockJudgeAgent:
    """Drop-in replacement for evaluator.llm.ClaudeAgent."""

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt
        self._rng = random.Random(_MOCK_SEED)

    def run(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_length: int = None,
        max_try: int = 5,
        success_check_fn: Callable = None,
    ):
        # Criterion names aren't passed to run() directly — BatchEvalAgent
        # bakes them into the prompt text via _format_criterion's "Name: X"
        # lines (batch_eval_prompt.py), so pull them back out from there.
        # Scoped to the text before "** Query **": the query/response the
        # criteria are judging are interpolated further down the same
        # prompt, and a response that happens to contain a line starting
        # with "Name: " (e.g. dialogue) would otherwise leak into the
        # extracted criteria names.
        criteria_section = prompt.split("** Query **", 1)[0]
        names = _NAME_RE.findall(criteria_section)
        response = json.dumps({
            name: {
                "score": self._rng.randint(1, 10),
                "reason": "mock judge (judge.type=mock) — no live API call",
            }
            for name in names
        })
        success = success_check_fn(response) if success_check_fn else True
        return response, success
