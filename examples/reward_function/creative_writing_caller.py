"""
Creative writing challenger reward function for VERL GRPO training.

Called by VERL at each training step with the challenger model's generated outputs.
Each predict string must contain a <output>...</output> JSON block with fields:
  - query:    the creative writing prompt
  - criteria: list of 5 evaluation criterion dicts

Reward pipeline:
  1. Parse all <output> JSONs (format validation)
  2. Batch-query vLLM solver (single POST with list of prompts)
  3. Score solver responses in parallel via BatchEvalAgent (Claude API)
  4. Compute R-Zero uncertainty via compute_writing_reward

Sampling parameters match solver_sampling.py (WritingBench defaults):
  temperature=0.7, top_p=0.8, top_k=20
  max_tokens defaults to 2048 for training speed (vs 16000 in full eval);
  override via CREATIVE_SOLVER_MAX_TOKENS env var.

Interface matches caller_penalty.py exactly:
  compute_score(predicts, ground_truths) -> List[Dict[str, float]]
  ground_truths is required by the VERL reward function interface but unused
  here — creative writing reward comes from the Claude judge, not a reference answer.
"""

import os
import sys
import requests
from typing import List, Dict, Optional
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure repo root and writing_bench dir are importable
_REPO = os.environ.get("REMOTE_REPO_PATH", "/root/R-Zero")
_WB_DIR = os.path.join(_REPO, "evaluation", "writing_bench")
for _p in (_REPO, _WB_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,0.0.0.0")

from question_generate.one_shot_creative_question_generate import FormatValidator, WritingPrompt
from evaluation.shared.rewards import compute_writing_reward

# Solver server config (GPU 7, started by creative_challenger_smoke.sh)
_SOLVER_PORT       = int(os.environ.get("CREATIVE_SOLVER_PORT", "5000"))
_SOLVER_MAX_TOKENS = int(os.environ.get("CREATIVE_SOLVER_MAX_TOKENS", "2048"))

# WritingBench sampling defaults (matches solver_sampling.py)
_TEMPERATURE = 0.7
_TOP_P       = 0.8
_TOP_K       = 20

# Lazy singletons
_agent = None
_solver_model_id = None


def _get_agent():
    global _agent
    if _agent is None:
        from evaluator import ClaudeAgent
        from batch_eval_agent import BatchEvalAgent
        from batch_eval_prompt import batch_evaluate_system
        _agent = BatchEvalAgent(ClaudeAgent(system_prompt=batch_evaluate_system))
    return _agent


def _get_solver_model_id() -> str:
    """Discover loaded model name from vLLM (cached after first call)."""
    global _solver_model_id
    if _solver_model_id is None:
        resp = requests.get(
            f"http://127.0.0.1:{_SOLVER_PORT}/v1/models", timeout=30
        )
        resp.raise_for_status()
        _solver_model_id = resp.json()["data"][0]["id"]
    return _solver_model_id


def _query_solver_batch(queries: List[str]) -> List[str]:
    """Send all queries in a single vLLM request; return completions in order."""
    resp = requests.post(
        f"http://127.0.0.1:{_SOLVER_PORT}/v1/completions",
        json={
            "model":       _get_solver_model_id(),
            "prompt":      queries,
            "max_tokens":  _SOLVER_MAX_TOKENS,
            "temperature": _TEMPERATURE,
            "top_p":       _TOP_P,
            "top_k":       _TOP_K,
        },
        timeout=300,
    )
    resp.raise_for_status()
    choices = sorted(resp.json()["choices"], key=lambda c: c["index"])
    return [c["text"] for c in choices]


def _score_one(
    agent,
    solver_text: str,
    prompt: WritingPrompt,
) -> Dict[str, float]:
    """Score one solver response against its criteria; returns reward dict."""
    try:
        criterion_scores = agent.score_all_criteria(
            content={"response": solver_text},
            query=prompt.query,
            criteria=prompt.criteria,
        )
        valid = [v["score"] for v in criterion_scores.values() if v.get("score", 0) > 0]
        avg   = mean(valid) if valid else 0.0
        return compute_writing_reward(avg_score=avg, format_valid=True)
    except Exception as e:
        print(f"[creative_writing_caller] scoring failed: {e}", flush=True)
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}


def compute_score(
    predicts: List[str],
    ground_truths: List[str],  # required by VERL interface; unused for creative writing
) -> List[Dict[str, float]]:
    """
    Compute R-Zero uncertainty rewards for challenger-generated creative prompts.

    Args:
        predicts:      Challenger model outputs (each must contain <output>…</output> JSON).
        ground_truths: Unused — required by VERL reward function interface.

    Returns:
        List of {"overall": float, "format": float, "accuracy": float} dicts.
    """
    agent = _get_agent()

    # --- Step 1: parse all predicts into WritingPrompt objects ---
    # validate_response checks XML tags, JSON validity, and that query is a str.
    # from_dict handles remaining field extraction with safe defaults.
    parsed: List[Optional[WritingPrompt]] = []
    for predict in predicts:
        fmt, p = FormatValidator.validate_response(predict)
        if fmt != 1:
            parsed.append(None)
            continue
        try:
            wp = WritingPrompt.from_dict(p)
        except Exception as e:
            print(f"[creative_writing_caller] WritingPrompt.from_dict failed: {e}", flush=True)
            parsed.append(None)
            continue
        parsed.append(wp if wp.query.strip() and wp.criteria else None)

    # --- Step 2: batch vLLM query for all valid predicts ---
    valid_idx     = [i for i, wp in enumerate(parsed) if wp is not None]
    valid_queries = [parsed[i].query for i in valid_idx]

    solver_texts: Dict[int, str] = {}
    if valid_queries:
        try:
            texts = _query_solver_batch(valid_queries)
            solver_texts = {idx: texts[k] for k, idx in enumerate(valid_idx)}
        except Exception as e:
            print(f"[creative_writing_caller] batch vLLM query failed: {e}", flush=True)

    # --- Step 3: parallel Claude scoring ---
    scores: List[Dict[str, float]] = [
        {"overall": 0.0, "format": 0.0, "accuracy": 0.0} for _ in predicts
    ]

    # Format valid but vLLM failed → give partial reward
    for idx in valid_idx:
        if idx not in solver_texts:
            scores[idx] = {"overall": 0.0, "format": 1.0, "accuracy": 0.0}

    # to_score entries: (list_index, solver_response_text, WritingPrompt)
    # _score_one receives the WritingPrompt and accesses .query / .criteria internally.
    to_score = [
        (idx, solver_texts[idx], parsed[idx])
        for idx in valid_idx
        if idx in solver_texts
    ]

    if to_score:
        with ThreadPoolExecutor(max_workers=min(len(to_score), 8)) as executor:
            futures = {
                executor.submit(_score_one, agent, solver_text, prompt): idx
                for idx, solver_text, prompt in to_score
            }
            for future in as_completed(futures):
                scores[futures[future]] = future.result()

    return scores
