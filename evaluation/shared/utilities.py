"""Shared utilities for the solver sampling / reward pipeline."""

from typing import Tuple


def split_thinking(text: str) -> Tuple[str, str]:
    """Split a thinking-mode completion into (thinking_trace, final_answer).

    Qwen3 thinking output looks like: "<think> …reasoning… </think>\n\n…answer".
    Only the final answer should be scored; the reasoning is stored separately.

    Three cases:
      • "</think>" present         → reasoning before it, answer after it.
      • "<think>" but no "</think>" → trace truncated (e.g. by max_tokens);
                                      it's all reasoning, no answer produced.
      • neither tag present         → model answered directly; whole text is
                                      the answer, nothing to strip.
    """
    close = "</think>"
    idx = text.find(close)
    if idx != -1:
        thinking = text[:idx]
        open_tag = thinking.find("<think>")
        if open_tag != -1:
            thinking = thinking[open_tag + len("<think>"):]
        return thinking.strip(), text[idx + len(close):].strip()

    if "<think>" in text:
        return text.strip(), ""

    return "", text.strip()
