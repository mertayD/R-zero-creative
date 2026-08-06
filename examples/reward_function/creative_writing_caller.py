"""
Creative writing challenger reward function for VERL GRPO training.

Called by VERL at each training step with the challenger model's generated outputs.
Each predict string must contain a <output>...</output> JSON block with fields:
  - query:    the creative writing prompt
  - criteria: list of 5 evaluation criterion dicts

Reward pipeline:
  1. Parse all <output> JSONs (format validation)
  2. Batch-query vLLM solver (single POST with list of prompts)
  3. Strip <think> traces; score only the final answers via BatchEvalAgent (Claude API)
  4. Compute R-Zero uncertainty via compute_writing_reward

Solver generation uses Qwen3 thinking-mode best practices:
  temperature=0.6, top_p=0.95, top_k=20, min_p=0 (no greedy decoding).
  Each query is wrapped in the solver chat template with enable_thinking=True
  before being sent to /v1/completions.
  max_tokens defaults to 32768 (thinking traces need a large budget);
  override via CREATIVE_SOLVER_MAX_TOKENS env var.

Interface matches caller_penalty.py exactly:
  compute_score(predicts, ground_truths) -> List[Dict[str, float]]
  ground_truths is required by the VERL reward function interface but unused
  here — creative writing reward comes from the Claude judge, not a reference answer.
"""

import json
import os
import sys
import requests
from typing import List, Dict, Optional, Tuple
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure repo root and writing_bench dir are importable
_REPO = os.environ.get("REMOTE_REPO_PATH", "/root/R-Zero")
_WB_DIR = os.path.join(_REPO, "evaluation", "writing_bench")
for _p in (_REPO, _WB_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,0.0.0.0")

from creative_rzero.data.writing_prompt import WritingPrompt
from creative_rzero.failure_reasons import FailureReason
from creative_rzero.utils import FormatValidator
from evaluation.shared.rewards import compute_writing_reward
from evaluation.shared.utilities import split_thinking
from batch_eval_agent import JudgeAPIError, JudgeParseError, JudgeInputError

# failure_reason taxonomy — every sample gets exactly one. Canonical values
# live in creative_rzero/failure_reasons.py:FailureReason —
#   ok                         — scored normally
#   truncated                  — <output>…</output> JSON never parsed, and the
#                                 response looks cut off by max_response_length
#                                 (model "couldn't finish", not "wrote garbage")
#   challenger_format_invalid  — <output>…</output> JSON never parsed, and it
#                                 does not look truncated
#   solver_api_error           — challenger parsed fine, but the vLLM solver
#                                 batch call failed (no solver_text to score)
#   empty_answer               — solver responded but with empty text
#   invalid_criteria           — wp.criteria was missing a required field
#                                 (data bug, not a judge failure)
#   judge_parse_fail           — judge responded every retry but never parsed
#   judge_rate_limit           — judge call failed with HTTP 429
#   judge_api_error            — judge call failed (network/5xx)

# Solver server config (GPU 7, started by creative_challenger_smoke.sh)
_SOLVER_PORT       = int(os.environ.get("CREATIVE_SOLVER_PORT", "5000"))
_SOLVER_MAX_TOKENS = int(os.environ.get("CREATIVE_SOLVER_MAX_TOKENS", "32768"))

# The challenger's own generation budget (worker.rollout / data.max_response_length
# in creative_challenger_smoke.sh) — used only to approximate truncation from
# decoded text, since compute_score never sees raw token counts.
_CHALLENGER_MAX_RESPONSE_LENGTH = int(os.environ.get("CHALLENGER_MAX_RESPONSE_LENGTH", "4096"))
_TRUNCATION_MARGIN_TOKENS = int(os.environ.get("TRUNCATION_MARGIN_TOKENS", "16"))
# Rough English/code heuristic — no tokenizer call per sample. Good enough to
# separate "ran out of budget" from "wrote malformed JSON with room to spare".
_APPROX_CHARS_PER_TOKEN = 3.5


def _looks_truncated(text: str, max_response_length: int) -> bool:
    """Best-effort truncation detector from decoded text alone.

    True when the response is within a few tokens' worth of characters of
    max_response_length, or an <output> block was opened but never closed.
    """
    approx_tokens = len(text) / _APPROX_CHARS_PER_TOKEN
    if approx_tokens >= max_response_length - _TRUNCATION_MARGIN_TOKENS:
        return True
    return "<output>" in text and "</output>" not in text

# Qwen3 thinking-mode sampling best practices (generation_config.json defaults).
# DO NOT use greedy decoding — it causes repetition/degradation in thinking mode.
_TEMPERATURE = 0.6
_TOP_P       = 0.95
_TOP_K       = 20
_MIN_P       = 0.0

# Lazy singletons
_agent = None
_solver_model_id = None
_solver_tokenizer = None
_wandb_run = None


def _get_wandb():
    global _wandb_run
    if _wandb_run is not None:
        return _wandb_run
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
        _wandb_run = wandb.init(
            id=os.environ.get("WANDB_RUN_ID") or None,
            resume="allow",
            reinit=True,
        )
    except Exception as e:
        print(f"[creative_writing_caller] W&B init failed: {e}", flush=True)
    return _wandb_run

# ---------------------------------------------------------------------------
# Per-rollout JSONL logger
# ---------------------------------------------------------------------------
# Appends one line per challenger rollout per compute_score call:
#   {STORAGE_PATH}/reward_logs/{VERL_EXPERIMENT_NAME}.jsonl
#
# Schema per line:
#   step                  — monotonic call counter
#   rollout_idx           — position in the batch
#   format_valid          — 1 if <output>JSON</output> parsed, 0 otherwise
#   challenger_thinking   — full <think> reasoning trace the challenger produced (empty if none)
#   generated_query       — first 300 chars of the writing prompt the challenger wrote
#   generated_criteria_n  — number of criteria in the generated prompt
#   solver_response       — first 300 chars of the solver's final answer (thinking stripped)
#   solver_thinking       — full <think> reasoning trace (empty if none / direct answer)
#   overall               — R-Zero uncertainty reward
#   format                — 1.0 = valid XML+JSON, 0.0 = parse failure
#   accuracy              — avg criterion score / 10
#
# The file is uploaded to W&B as a Table by creative_challenger_smoke.sh after
# training completes.
# ---------------------------------------------------------------------------
_challenger_step: int = 0


def _log_challenger_rollouts(
    parsed: List[Optional[WritingPrompt]],
    challenger_thinking: List[str],
    solver_texts: Dict[int, str],
    solver_thinking: Dict[int, str],
    scores: List[Dict[str, float]],
    failure_reasons: List[str],
) -> None:
    global _challenger_step
    _challenger_step += 1

    storage  = os.environ.get("STORAGE_PATH", "/tmp")
    exp_name = os.environ.get("VERL_EXPERIMENT_NAME", "unknown_experiment")
    log_dir  = os.path.join(storage, "reward_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp_name}.jsonl")

    entries: List[str] = []
    for idx, wp in enumerate(parsed):
        entry = {
            "step":                 _challenger_step,
            "rollout_idx":          idx,
            "format_valid":         1 if wp is not None else 0,
            "challenger_thinking":  challenger_thinking[idx],
            "generated_query":      wp.query[:300] if wp is not None else "",
            "generated_criteria_n": len(wp.criteria) if wp is not None else 0,
            "solver_response":      solver_texts.get(idx, "")[:300],
            "solver_thinking":      solver_thinking.get(idx, ""),
            "overall":              round(scores[idx].get("overall", 0.0), 4),
            "format":               round(scores[idx].get("format",  0.0), 4),
            "accuracy":             round(scores[idx].get("accuracy", 0.0), 4),
            "failure_reason":       failure_reasons[idx],
        }
        entries.append(json.dumps(entry))

    if entries:
        with open(log_path, "a") as f:
            f.write("\n".join(entries) + "\n")


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


def _get_solver_tokenizer():
    """Load the solver's HF tokenizer (cached after first call)."""
    global _solver_tokenizer
    if _solver_tokenizer is None:
        from transformers import AutoTokenizer
        _solver_tokenizer = AutoTokenizer.from_pretrained(_get_solver_model_id())
    return _solver_tokenizer


def _build_thinking_prompt(query: str) -> str:
    """Wrap a query in the solver chat template with thinking mode enabled.

    Applied client-side so we can keep sending a batch of prompts to the
    /v1/completions endpoint in a single request.
    """
    tokenizer = _get_solver_tokenizer()
    messages = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # True is the default value for enable_thinking
    )


def _query_solver_batch(queries: List[str]) -> List[str]:
    """Send all queries in a single vLLM request; return completions in order."""
    prompts = [_build_thinking_prompt(q) for q in queries]
    resp = requests.post(
        f"http://127.0.0.1:{_SOLVER_PORT}/v1/completions",
        json={
            "model":       _get_solver_model_id(),
            "prompt":      prompts,
            "max_tokens":  _SOLVER_MAX_TOKENS,
            "temperature": _TEMPERATURE,
            "top_p":       _TOP_P,
            "top_k":       _TOP_K,
            "min_p":       _MIN_P,
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
) -> Tuple[Dict[str, float], str]:
    """Score one solver response against its criteria.

    Returns (reward_dict, failure_reason). reward_dict's overall/accuracy are
    0.0 whenever failure_reason != FailureReason.OK.
    """
    if not solver_text.strip():
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}, FailureReason.EMPTY_ANSWER.value
    try:
        criterion_scores = agent.score_all_criteria(
            content={"response": solver_text},
            query=prompt.query,
            criteria=prompt.criteria,
        )
    except JudgeInputError as e:
        print(f"[creative_writing_caller] invalid_criteria: {e}", flush=True)
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}, FailureReason.INVALID_CRITERIA.value
    except JudgeAPIError as e:
        reason = FailureReason.JUDGE_RATE_LIMIT.value if e.status_code == 429 else FailureReason.JUDGE_API_ERROR.value
        print(f"[creative_writing_caller] {reason}: {e}", flush=True)
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}, reason
    except JudgeParseError as e:
        print(f"[creative_writing_caller] judge_parse_fail: {e}", flush=True)
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}, FailureReason.JUDGE_PARSE_FAIL.value
    except Exception as e:
        print(f"[creative_writing_caller] scoring failed: {e}", flush=True)
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}, FailureReason.JUDGE_API_ERROR.value

    valid = [v["score"] for v in criterion_scores.values() if v.get("score", 0) > 0]
    if not valid:
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}, FailureReason.JUDGE_PARSE_FAIL.value
    avg = mean(valid)
    return compute_writing_reward(avg_score=avg, format_valid=True), FailureReason.OK.value


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
    # Challenger rollouts are thinking-mode: split off the <think>…</think>
    # trace first (logged separately), then validate the answer. validate_response
    # checks XML tags, JSON validity, and that query is a str; from_dict handles
    # remaining field extraction with safe defaults.
    parsed: List[Optional[WritingPrompt]] = []
    challenger_thinking: List[str] = []
    failure_reasons: List[str] = [FailureReason.OK.value] * len(predicts)
    for i, predict in enumerate(predicts):
        thinking, answer = split_thinking(predict)
        challenger_thinking.append(thinking)
        fmt, p = FormatValidator.validate_response(answer)
        if fmt != 1:
            parsed.append(None)
            failure_reasons[i] = (
                FailureReason.TRUNCATED.value
                if _looks_truncated(predict, _CHALLENGER_MAX_RESPONSE_LENGTH)
                else FailureReason.CHALLENGER_FORMAT_INVALID.value
            )
            continue
        try:
            wp = WritingPrompt.from_dict(p)
        except Exception as e:
            print(f"[creative_writing_caller] WritingPrompt.from_dict failed: {e}", flush=True)
            parsed.append(None)
            failure_reasons[i] = FailureReason.CHALLENGER_FORMAT_INVALID.value
            continue
        if wp.query.strip() and wp.criteria:
            parsed.append(wp)
        else:
            parsed.append(None)
            failure_reasons[i] = FailureReason.CHALLENGER_FORMAT_INVALID.value

    # --- Step 2: batch vLLM query for all valid predicts ---
    valid_idx     = [i for i, wp in enumerate(parsed) if wp is not None]
    valid_queries = [parsed[i].query for i in valid_idx]

    # solver_texts holds the final answers (thinking stripped) used for scoring;
    # solver_thinking holds the reasoning traces, logged separately.
    solver_texts: Dict[int, str] = {}
    solver_thinking: Dict[int, str] = {}
    if valid_queries:
        try:
            texts = _query_solver_batch(valid_queries)
            for k, idx in enumerate(valid_idx):
                thinking, answer = split_thinking(texts[k])
                solver_texts[idx]    = answer
                solver_thinking[idx] = thinking
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
            failure_reasons[idx] = FailureReason.SOLVER_API_ERROR.value

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
                idx = futures[future]
                scores[idx], failure_reasons[idx] = future.result()

    try:
        _log_challenger_rollouts(
            parsed, challenger_thinking, solver_texts, solver_thinking, scores, failure_reasons
        )
    except Exception as _log_err:
        print(f"[creative_writing_caller] rollout logging failed: {_log_err}", flush=True)

    try:
        wb = _get_wandb()
        if wb is not None:
            n_total   = len(predicts)
            n_valid   = len(valid_idx)
            n_solved  = len(solver_texts)
            valid_scores   = [scores[i] for i in valid_idx]
            overall_vals   = [s["overall"]  for s in valid_scores] if valid_scores else [0.0]
            accuracy_vals  = [s["accuracy"] for s in valid_scores] if valid_scores else [0.0]
            criteria_counts = [
                len(parsed[i].criteria) for i in valid_idx if parsed[i] is not None
            ]
            log_dict = {
                "challenger/format_valid_rate":   n_valid  / n_total  if n_total  else 0.0,
                "challenger/solver_success_rate": n_solved / n_valid  if n_valid  else 0.0,
                "challenger/mean_overall_reward": mean(overall_vals),
                "challenger/mean_accuracy":       mean(accuracy_vals),
                "challenger/mean_criteria_count": mean(criteria_counts) if criteria_counts else 0.0,
                "challenger/num_valid":           n_valid,
                "challenger/num_samples":         n_total,
            }
            reason_counts: Dict[str, int] = {}
            for reason in failure_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in reason_counts.items():
                log_dict[f"challenger/failure_reason/{reason}"] = count

            wb.log(log_dict, step=_challenger_step)
    except Exception as _wb_err:
        print(f"[creative_writing_caller] W&B logging failed: {_wb_err}", flush=True)

    return scores
