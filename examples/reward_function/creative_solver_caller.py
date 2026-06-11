"""
Creative writing solver reward function — Pairwise Tournament Normalized Rank.

Called by VERL at each GRPO training step with the solver model's rollout outputs.
Each predict is a raw creative writing response (plain text, no required format).

Data flow
---------
  Training parquet:  problem = writing task query  (what the solver sees as input)
                     answer  = WritingPrompt.to_dict() JSON  (passed as ground_truth)

  Parquet is pre-validated: every row is guaranteed to have a non-empty query and
  at least one evaluation criterion.  See creative_solver_smoke.sh Step 1.

  VERL rollout:      for each prompt, solver generates G independent responses.
                     worker.rollout.n = G controls this; G is not hardcoded here.

  Reward fn receives:
    predicts       = [resp_p0_r0, ..., resp_p0_r{G-1},
                      resp_p1_r0, ..., resp_p{N-1}_r{G-1}]
    ground_truths  = [wp_p0_json × G, wp_p1_json × G, ..., wp_p{N-1}_json × G]

  Reward fn does:
    1. Parse each ground_truth with WritingPrompt.from_dict → prompt_id, query,
       criteria.  Data is pre-validated; any parse failure is a hard error.
    2. Group flat list into N prompt groups keyed by prompt_id.
       G_eff = len(group) — adapts if a rollout failed to generate (M → M-1).
    3. Score all responses in parallel with BatchEvalAgent (one Claude call each).
       Concurrency capped by CREATIVE_SCORER_MAX_WORKERS (default 4) to stay
       within the Claude API rate limit (~50 RPM).
    4. Within each group rank by avg criterion score and assign normalised rank:

           R_i = (G_eff - rank_i) / (G_eff - 1)    rank_i ∈ {1 … G_eff}  (1 = best)

       Range [0,1], mean 0.5 → GRPO advantages are always zero-centred.

Return format per sample:
    {"overall": rank_reward, "format": 1.0, "accuracy": avg_score / 10.0}
VERL uses `overall` for the GRPO update; `accuracy` is logged as a metric.
"""

import json
import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Dict, List, Tuple

# Ensure repo root and writing_bench dir are importable
_REPO   = os.environ.get("REMOTE_REPO_PATH", "/root/R-Zero")
_WB_DIR = os.path.join(_REPO, "evaluation", "writing_bench")
for _p in (_REPO, _WB_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,0.0.0.0")

from question_generate.one_shot_creative_question_generate import WritingPrompt

# Concurrent Claude calls — keep below ~50 RPM rate limit.
# Raise via CREATIVE_SCORER_MAX_WORKERS if your API tier allows more.
_MAX_WORKERS = int(os.environ.get("CREATIVE_SCORER_MAX_WORKERS", "4"))

# ---------------------------------------------------------------------------
# Per-rollout JSONL logger
# ---------------------------------------------------------------------------
# Appends one line per rollout per compute_score call to a sidecar file:
#   {STORAGE_PATH}/reward_logs/{VERL_EXPERIMENT_NAME}.jsonl
#
# Schema per line:
#   step          — monotonic call counter (not VERL's global step, but
#                   proportional to it: one increment per compute_score call)
#   prompt_id     — WritingPrompt.prompt_id (from ground_truth)
#   domain        — wp.domain  (e.g. "D1")
#   domain_name   — wp.domain_name
#   subdomain     — wp.subdomain
#   num_criteria  — len(wp.criteria)
#   response_preview — first 400 chars of the solver's rollout response
#   raw_score     — avg criterion score 1–10 from Claude
#   rank_reward   — normalised rank reward [0, 1] (GRPO signal)
#   accuracy      — raw_score / 10  (logged metric)
#
# The file is uploaded to W&B as a Table by creative_solver_smoke.sh after
# training completes.
# ---------------------------------------------------------------------------
_solver_step: int = 0


def _log_solver_rollouts(
    groups: "OrderedDict[str, dict]",
    raw_scores: List[float],
    rewards: List[Dict[str, float]],
) -> None:
    global _solver_step
    _solver_step += 1

    storage  = os.environ.get("STORAGE_PATH", "/tmp")
    exp_name = os.environ.get("VERL_EXPERIMENT_NAME", "unknown_experiment")
    log_dir  = os.path.join(storage, "reward_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp_name}.jsonl")

    entries: List[str] = []
    for group in groups.values():
        wp: WritingPrompt = group["wp"]
        for idx, response_text in group["samples"]:
            entries.append(json.dumps({
                "step":             _solver_step,
                "prompt_id":        wp.prompt_id,
                "domain":           wp.domain,
                "domain_name":      wp.domain_name,
                "subdomain":        wp.subdomain,
                "num_criteria":     len(wp.criteria),
                "input_query":      wp.query,
                "response_preview": response_text[:400],
                "raw_score":        round(raw_scores[idx], 4),
                "rank_reward":      round(rewards[idx].get("overall", 0.0), 4),
                "accuracy":         round(rewards[idx].get("accuracy", 0.0), 4),
            }))

    if entries:
        with open(log_path, "a") as f:
            f.write("\n".join(entries) + "\n")

# ---------------------------------------------------------------------------
# Lazy singleton — BatchEvalAgent imported only on first call
# ---------------------------------------------------------------------------
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from evaluator import ClaudeAgent
        from batch_eval_agent import BatchEvalAgent
        from batch_eval_prompt import batch_evaluate_system
        _agent = BatchEvalAgent(ClaudeAgent(system_prompt=batch_evaluate_system))
    return _agent


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------

def _score_one(agent, response_text: str, wp: WritingPrompt) -> float:
    """Score one solver response against all WritingPrompt criteria; return avg (1–10)."""
    try:
        criterion_scores = agent.score_all_criteria(
            content={"response": response_text},
            query=wp.query,
            criteria=wp.criteria,
        )
        valid = [
            v["score"]
            for v in criterion_scores.values()
            if isinstance(v.get("score"), (int, float)) and v["score"] > 0
        ]
        return mean(valid) if valid else 0.0
    except Exception as e:
        print(f"[creative_solver_caller] scoring failed: {e}", flush=True)
        return 0.0


# ---------------------------------------------------------------------------
# Normalised rank reward
# ---------------------------------------------------------------------------

def _assign_normalised_rank_rewards(
    eval_scores: Dict[int, float],
) -> Dict[int, float]:
    """
    Convert {sample_idx: avg_score} into normalised rank rewards.

    Uses G_eff = len(eval_scores) so the formula adapts when some rollouts
    fail to generate (M requested → M-1 produced).

        R_i = (G_eff - rank_i) / (G_eff - 1)    rank_i starts at 1 (best)

    G_eff == 1 → {idx: 0.5}  (neutral, no signal, avoids division by zero).
    """
    G_eff = len(eval_scores)

    if G_eff == 0:
        return {}

    if G_eff == 1:
        idx = next(iter(eval_scores))
        return {idx: 0.5}

    rewards: Dict[int, float] = {}

    # Sort by score descending (best first)
    sorted_samples: List[Tuple[int, float]] = sorted(
        eval_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Assign normalised rank — rank starts at 1
    for rank, (sample_idx, score) in enumerate(sorted_samples, start=1):
        normalized_reward = (G_eff - rank) / (G_eff - 1)
        rewards[sample_idx] = round(normalized_reward, 4)

    return rewards


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_score(
    predicts: List[str],
    ground_truths: List[str],
) -> List[Dict[str, float]]:
    """
    Compute per-sample normalised rank rewards for VERL GRPO solver training.

    Args:
        predicts:      Flat list of solver writing responses (length N × G).
        ground_truths: WritingPrompt.to_dict() JSON strings, same value repeated
                       G times per prompt (VERL repeats the answer column once
                       per rollout). Pre-validated at parquet build time.

    Returns:
        List[Dict] aligned with predicts:
          overall  — rank reward ∈ [0, 1]       (GRPO update signal)
          format   — always 1.0 (data pre-screened at parquet build)
          accuracy — avg criterion score / 10   (logging metric)
    """
    agent     = _get_agent()
    n_samples = len(predicts)

    # ------------------------------------------------------------------ #
    # Step 1 — parse ground_truths and group by prompt_id                 #
    # ------------------------------------------------------------------ #
    # groups: prompt_id -> {"wp": WritingPrompt, "samples": [(idx, pred)]}
    # Data is pre-validated at parquet build; WritingPrompt.from_dict raises
    # on bad input — that is the intended behaviour here (fail loud, fail early).
    groups: "OrderedDict[str, dict]" = OrderedDict()

    for i, (pred, gt) in enumerate(zip(predicts, ground_truths)):
        wp  = WritingPrompt.from_dict(json.loads(gt))
        key = wp.prompt_id

        if key not in groups:
            groups[key] = {"wp": wp, "samples": []}
        groups[key]["samples"].append((i, pred))

    # ------------------------------------------------------------------ #
    # Step 2 — score all samples in parallel                              #
    # ------------------------------------------------------------------ #
    # raw_scores[i] = avg criterion score (1–10) for predicts[i]
    raw_scores: List[float] = [0.0] * n_samples

    tasks: List[Tuple[int, str, WritingPrompt]] = [
        (idx, pred, group["wp"])
        for group in groups.values()
        for idx, pred in group["samples"]
    ]

    # Cap concurrency to stay within Claude API rate limits.
    # CREATIVE_SCORER_MAX_WORKERS defaults to 4; raise if your tier allows more.
    max_workers = min(_MAX_WORKERS, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_score_one, agent, pred, wp): idx
            for idx, pred, wp in tasks
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                raw_scores[idx] = future.result()
            except Exception as e:
                print(
                    f"[creative_solver_caller] future failed idx={idx}: {e}",
                    flush=True,
                )
                raw_scores[idx] = 0.0

    # ------------------------------------------------------------------ #
    # Step 3 — per-group normalised rank rewards                          #
    # ------------------------------------------------------------------ #
    rewards: List[Dict[str, float]] = [{} for _ in range(n_samples)]

    for group in groups.values():
        # G_eff = actual number of rollouts received for this prompt
        eval_scores_group: Dict[int, float] = {
            idx: raw_scores[idx]
            for idx, _ in group["samples"]
        }

        rank_rewards = _assign_normalised_rank_rewards(eval_scores_group)

        for idx, rank_r in rank_rewards.items():
            rewards[idx] = {
                "overall":  rank_r,
                "format":   1.0,
                "accuracy": round(raw_scores[idx] / 10.0, 4),
            }

    # ------------------------------------------------------------------ #
    # Step 4 — diagnostic log (mean should be ≈ 0.500)                   #
    # ------------------------------------------------------------------ #
    overall_values = [r["overall"] for r in rewards]
    print(
        f"[creative_solver_caller] "
        f"groups={len(groups)}  samples={n_samples}  workers={max_workers}  "
        f"mean_rank_reward={mean(overall_values):.3f}  (expected≈0.500)",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # Step 5 — per-rollout JSONL logging                                  #
    # ------------------------------------------------------------------ #
    try:
        _log_solver_rollouts(groups, raw_scores, rewards)
    except Exception as _log_err:
        print(f"[creative_solver_caller] rollout logging failed: {_log_err}", flush=True)

    return rewards
