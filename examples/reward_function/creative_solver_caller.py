"""
Creative writing solver reward function — judge scoring + orchestration,
with the reward formula itself pluggable via creative_rzero/rewards/.

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
    4. Turn each group's avg criterion scores into `overall` by delegating
       to a pluggable reward strategy (creative_rzero/rewards/registry.py,
       T3.3), selected by CREATIVE_SOLVER_REWARD_TYPE (bridged from
       rewards.solver.type — see verl_entry.py::_bridge_config_to_env).
       Two strategies exist today (creative_rzero/rewards/solver/):
         rank (default) — normalised within-group rank, see rank.py
         raw            — judge's own score rescaled to [0,1], see raw.py
       Each strategy is self-contained and independently unit-testable
       against a GroupScores fixture — no judge, no env vars. Adding a new
       formula (zscore, pairwise, ...) means a new strategy file plus one
       line in an experiment config, not a new branch here.

Return format per sample:
    {"overall": <rank or raw reward>, "format": 1.0, "accuracy": avg_score / 10.0}
VERL uses `overall` for the GRPO update; `accuracy` is logged as a metric
regardless of reward type, so the two modes stay comparable on W&B.
"""

import json
import os
import sys
from collections import OrderedDict, defaultdict
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

from creative_rzero.data.writing_prompt import WritingPrompt
from creative_rzero.failure_reasons import FailureReason
from creative_rzero.rewards.registry import GroupScores
from creative_rzero.utils import find_cjk_matches
from evaluation.shared.utilities import split_thinking
from batch_eval_agent import JudgeAPIError, JudgeParseError, JudgeInputError

# Concurrent Claude calls — keep below ~50 RPM rate limit.
# Raise via CREATIVE_SCORER_MAX_WORKERS if your API tier allows more.
_MAX_WORKERS = int(os.environ.get("CREATIVE_SCORER_MAX_WORKERS", "4"))

# Solver responses have no required format (format is always 1.0 for
# pre-validated data), so a truncated response is still scoreable — it just
# might read as truncated to the judge. Tracked as an additive `truncated`
# flag alongside failure_reason rather than a failure_reason value of its
# own, since it does not by itself mean the sample was not scored.
_SOLVER_MAX_RESPONSE_LENGTH = int(os.environ.get("SOLVER_MAX_RESPONSE_LENGTH", "4096"))
_TRUNCATION_MARGIN_TOKENS = int(os.environ.get("TRUNCATION_MARGIN_TOKENS", "16"))
_APPROX_CHARS_PER_TOKEN = 3.5  # rough heuristic — no tokenizer call per sample


def _looks_truncated(text: str, max_response_length: int) -> bool:
    """Best-effort truncation detector from decoded text alone (character
    length as a proxy for token count, since compute_score never sees raw
    token counts)."""
    approx_tokens = len(text) / _APPROX_CHARS_PER_TOKEN
    return approx_tokens >= max_response_length - _TRUNCATION_MARGIN_TOKENS

# Two independent, configurable *definitions* — computed once here, shared
# by every reward strategy so they can't disagree about which groups they
# describe. What a strategy does in response is its own decision (see
# creative_rzero/rewards/solver/{rank,raw}.py) — this block only decides
# whether each one applies:
#   all_failed   — every scoreable sample in the group has failure_reason !=
#                  "ok" (nothing to score at all).
#   low_quality  — group scored fine, but its best raw score (1-10 scale) is
#                  still at or below _LOW_QUALITY_THRESHOLD (indistinguishable
#                  low-quality content, not a scoring failure).
# The old `_MIN_SCORE = 0.3` compared against a 1-10 scale — i.e. it could
# only ever fire when every sample was already a hard failure (raw_score
# exactly 0.0), so it was accidentally doing all_failed's job while being
# undocumented as a "low quality" gate that never actually fired.
_LOW_QUALITY_THRESHOLD = float(os.environ.get("CREATIVE_LOW_QUALITY_THRESHOLD", "2.0"))

# Selects the reward strategy from creative_rzero/rewards/registry.py —
# "rank" (default) or "raw" today, see module docstring step 4. Bridged
# from rewards.solver.type by verl_entry.py::_bridge_config_to_env;
# resolved (and validated) lazily by _get_strategy() below, same pattern
# as WB_JUDGE_TYPE in _get_agent().
_REWARD_TYPE = os.environ.get("CREATIVE_SOLVER_REWARD_TYPE", "rank")
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
#   failure_reason — see taxonomy above _score_one
#   truncated     — response length is within a few tokens of
#                   SOLVER_MAX_RESPONSE_LENGTH (best-effort, char-based proxy)
#   is_low_quality — group-level flag: every scoreable sample in this
#                   prompt's group scored <= _LOW_QUALITY_THRESHOLD (the
#                   shared definition above). What happens to the reward as
#                   a result is up to the active strategy — RankReward
#                   overrides to a uniform 0.5, RawReward leaves it as-is
#                   (see creative_rzero/rewards/solver/{rank,raw}.py) — so
#                   this flag is bookkeeping regardless of mode. Distinguishes
#                   "scored fine but uniformly bad content" from a genuine
#                   all_failed collapse (rank_reward == 0.0), since both
#                   would otherwise look identical to a consumer only
#                   checking failure_reason. See select_checkpoint.py.
#   rank_reward   — the `overall` GRPO reward produced by the active
#                   strategy — despite the field name, holds a rank reward
#                   [0, 1] in rank mode or a rescaled raw score [0, 1] in raw
#                   mode. Field name kept as-is: creative_rzero/steps/report.py
#                   reads this key for its solver-reward fallback regardless
#                   of which strategy produced it.
#   accuracy      — raw_score / 10  (logged metric, same in both modes)
#
# The file is uploaded to W&B as a Table by creative_solver_smoke.sh after
# training completes.
# ---------------------------------------------------------------------------
_solver_step: int = 0


def _log_solver_rollouts(
    groups: "OrderedDict[str, dict]",
    raw_scores: List[float],
    failure_reasons: List[str],
    truncated_flags: List[bool],
    rewards: List[Dict[str, float]],
    low_quality_groups: Dict[str, bool],
) -> None:
    global _solver_step
    _solver_step += 1

    storage  = os.environ.get("STORAGE_PATH", "/tmp")
    exp_name = os.environ.get("VERL_EXPERIMENT_NAME", "unknown_experiment")
    log_dir  = os.path.join(storage, "reward_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp_name}.jsonl")

    entries: List[str] = []
    for prompt_id, group in groups.items():
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
                "failure_reason":   failure_reasons[idx],
                "truncated":        truncated_flags[idx],
                "is_low_quality": low_quality_groups.get(prompt_id, False),
                "rank_reward":      round(rewards[idx].get("overall", 0.0), 4),
                "accuracy":         round(rewards[idx].get("accuracy", 0.0), 4),
            }))

    if entries:
        with open(log_path, "a") as f:
            f.write("\n".join(entries) + "\n")

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_agent = None
_wandb_run = None
_strategy = None


def _get_strategy():
    """Resolve _REWARD_TYPE to a RewardStrategy instance via the T3.3
    registry (creative_rzero/rewards/registry.py). Importing
    creative_rzero.rewards.solver is what registers "rank"/"raw" — done
    here, lazily, so a bad CREATIVE_SOLVER_REWARD_TYPE fails loud on first
    use instead of at module import."""
    global _strategy
    if _strategy is None:
        from creative_rzero.rewards.registry import get_strategy
        import creative_rzero.rewards.solver  # noqa: F401 — registers rank/raw strategies

        try:
            strategy_cls = get_strategy(_REWARD_TYPE)
        except KeyError as e:
            raise ValueError(
                f"CREATIVE_SOLVER_REWARD_TYPE={_REWARD_TYPE!r} is not a recognized solver "
                f"reward type ({e}). It should be set by verl_entry.py::_bridge_config_to_env "
                "from rewards.solver.type in the experiment config — check "
                "EXPERIMENT_CONFIG_PATH if this fired unexpectedly."
            ) from None
        _strategy = strategy_cls()
    return _strategy


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
        print(f"[creative_solver_caller] W&B init failed: {e}", flush=True)
    return _wandb_run


def _get_agent():
    global _agent
    if _agent is None:
        from batch_eval_agent import BatchEvalAgent, PerCriterionEvalAgent
        from batch_eval_prompt import batch_evaluate_system
        judge_type = os.environ.get("WB_JUDGE_TYPE")
        if judge_type == "mock":
            from evaluator import MockJudgeAgent
            judge = MockJudgeAgent(system_prompt=batch_evaluate_system)
            _agent = BatchEvalAgent(judge)
        elif judge_type == "sft-critic":
            # sft-critic scores per-criterion, not batched — see
            # PerCriterionEvalAgent's docstring and REFACTOR_PLAN.md §6.3/§6.4
            # for why this backend does not reuse batch_evaluate_system/
            # BatchEvalAgent the way claude/mock do.
            from evaluator import CriticServerAgent
            from per_criterion_eval_prompt import evaluate_system as per_criterion_evaluate_system
            judge = CriticServerAgent(system_prompt=per_criterion_evaluate_system)
            _agent = PerCriterionEvalAgent(judge)
        elif judge_type == "claude":
            from evaluator import ClaudeAgent
            judge = ClaudeAgent(system_prompt=batch_evaluate_system)
            _agent = BatchEvalAgent(judge)
        else:
            # No silent default to claude: an unset/typo'd/unrecognized
            # WB_JUDGE_TYPE must fail loud here, not quietly burn Claude API
            # calls under the wrong judge. creative_rzero/config.py already
            # validates judge.type at config-load time (VALID_JUDGE_TYPES) —
            # this is defense-in-depth for direct/manual invocation, not the
            # primary guard.
            raise ValueError(
                f"WB_JUDGE_TYPE={judge_type!r} is not a recognized judge type "
                "(expected one of: claude, mock, sft-critic). It should be set "
                "by verl_entry.py::_bridge_config_to_env from judge.type in the "
                "experiment config — check EXPERIMENT_CONFIG_PATH if this fired "
                "unexpectedly."
            )
    return _agent


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------
# failure_reason taxonomy — every sample gets exactly one, so distributions
# like "79% exact zero" become a one-groupby answer instead of a mystery.
# Canonical values live in creative_rzero/failure_reasons.py:FailureReason —
#   ok                — scored normally
#   empty_answer      — response text was empty, nothing to grade
#   invalid_criteria  — wp.criteria was missing a required field (data bug,
#                        not a judge failure) — fails before any judge call
#   judge_parse_fail  — judge responded every retry but output never parsed
#   judge_rate_limit  — judge call failed with HTTP 429 after its own retries
#   judge_api_error   — judge call failed (network/5xx) after its own retries
#   language_filter   — non-English response, excluded before scoring
# ---------------------------------------------------------------------------

def _score_one(agent, response_text: str, wp: WritingPrompt) -> Tuple[float, str]:
    """Score one solver response against all WritingPrompt criteria.

    Returns (avg_score_1_to_10, failure_reason). avg_score is 0.0 whenever
    failure_reason != FailureReason.OK.
    """
    if not response_text.strip():
        return 0.0, FailureReason.EMPTY_ANSWER.value
    try:
        criterion_scores = agent.score_all_criteria(
            content={"response": response_text},
            query=wp.query,
            criteria=wp.criteria,
        )
    except JudgeInputError as e:
        print(f"[creative_solver_caller] invalid_criteria: {e}", flush=True)
        return 0.0, FailureReason.INVALID_CRITERIA.value
    except JudgeAPIError as e:
        reason = FailureReason.JUDGE_RATE_LIMIT.value if e.status_code == 429 else FailureReason.JUDGE_API_ERROR.value
        print(f"[creative_solver_caller] {reason}: {e}", flush=True)
        return 0.0, reason
    except JudgeParseError as e:
        print(f"[creative_solver_caller] judge_parse_fail: {e}", flush=True)
        return 0.0, FailureReason.JUDGE_PARSE_FAIL.value
    except Exception as e:
        print(f"[creative_solver_caller] scoring failed: {e}", flush=True)
        return 0.0, FailureReason.JUDGE_API_ERROR.value

    valid = [
        v["score"]
        for v in criterion_scores.values()
        if isinstance(v.get("score"), (int, float)) and v["score"] > 0
    ]
    if not valid:
        return 0.0, FailureReason.JUDGE_PARSE_FAIL.value
    return mean(valid), FailureReason.OK.value


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_score(
    predicts: List[str],
    ground_truths: List[str],
) -> List[Dict[str, float]]:
    """
    Compute per-sample solver rewards for VERL GRPO training, via the
    active reward strategy (CREATIVE_SOLVER_REWARD_TYPE — see
    creative_rzero/rewards/registry.py and _get_strategy() above).

    Args:
        predicts:      Flat list of solver writing responses (length N × G).
        ground_truths: WritingPrompt.to_dict() JSON strings, same value repeated
                       G times per prompt (VERL repeats the answer column once
                       per rollout). Pre-validated at parquet build time.

    Returns:
        List[Dict] aligned with predicts:
          overall  — reward ∈ [0, 1] from the active strategy (GRPO update signal)
          format   — always 1.0 (data pre-screened at parquet build)
          accuracy — avg criterion score / 10   (logging metric)
    """
    strategy  = _get_strategy()
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

    truncated_flags: List[bool] = [
        _looks_truncated(pred, _SOLVER_MAX_RESPONSE_LENGTH) for pred in predicts
    ]

    # ------------------------------------------------------------------ #
    # Step 1b — language filter: skip non-English solver responses        #
    # ------------------------------------------------------------------ #
    # Filters on the answer only (thinking traces can legitimately contain
    # stray non-English tokens without the final answer being non-English).
    # No-op today since thinking is disabled for the solver — split_thinking
    # returns the whole text as the answer — but this is correct the day
    # apply_chat_template_kwargs.enable_thinking is turned on (see T1.10).
    failure_reasons: List[str] = [""] * n_samples
    language_filtered: set = set()
    for i, pred in enumerate(predicts):
        _, pred_answer = split_thinking(pred)
        cjk_matches = find_cjk_matches(pred_answer)
        if cjk_matches:
            language_filtered.add(i)
            failure_reasons[i] = FailureReason.LANGUAGE_FILTER.value
            print(
                f"[creative_solver_caller] non-English answer at idx={i} "
                f"({len(cjk_matches)} CJK chars matched) — assigning zero reward; "
                f"preview={pred_answer[:200]!r}",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # Step 2 — score all samples in parallel                              #
    # ------------------------------------------------------------------ #
    # raw_scores[i] = avg criterion score (1–10) for predicts[i]
    raw_scores: List[float] = [0.0] * n_samples

    # Judge on the answer only — same reasoning as the T1.8 language filter.
    # No-op today (thinking disabled, split_thinking returns the whole text
    # as the answer); once enable_thinking is turned on this also means a
    # thinking-only, no-answer rollout correctly falls through _score_one's
    # empty-text check to "empty_answer" instead of scoring the reasoning
    # trace as if it were the response.
    tasks: List[Tuple[int, str, WritingPrompt]] = [
        (idx, split_thinking(pred)[1], group["wp"])
        for group in groups.values()
        for idx, pred in group["samples"]
        if idx not in language_filtered
    ]

    # Cap concurrency to stay within Claude API rate limits.
    # CREATIVE_SCORER_MAX_WORKERS defaults to 4; raise if your tier allows more.
    if tasks:
        max_workers = min(_MAX_WORKERS, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_score_one, agent, pred, wp): idx
                for idx, pred, wp in tasks
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    raw_scores[idx], failure_reasons[idx] = future.result()
                except Exception as e:
                    print(
                        f"[creative_solver_caller] future failed idx={idx}: {e}",
                        flush=True,
                    )
                    raw_scores[idx] = 0.0
                    failure_reasons[idx] = FailureReason.JUDGE_API_ERROR.value
    else:
        max_workers = 0

    # ------------------------------------------------------------------ #
    # Step 3 — per-group rewards, via the active strategy                 #
    # ------------------------------------------------------------------ #
    rewards: List[Dict[str, float]] = [{} for _ in range(n_samples)]
    n_all_failed_groups: int = 0
    n_low_quality_groups: int = 0
    low_quality_groups: Dict[str, bool] = {}

    for prompt_id, group in groups.items():
        # G_eff = actual number of rollouts received for this prompt
        eval_scores_group: Dict[int, float] = {
            idx: raw_scores[idx]
            for idx, _ in group["samples"]
        }

        # Exclude language-filtered samples from ranking so they don't distort
        # the rank distribution for valid responses in the same group.
        scoreable = {idx: s for idx, s in eval_scores_group.items() if idx not in language_filtered}

        all_failed = bool(scoreable) and all(failure_reasons[idx] != FailureReason.OK for idx in scoreable)
        low_quality = (
            bool(scoreable) and not all_failed and max(scoreable.values()) <= _LOW_QUALITY_THRESHOLD
        )
        low_quality_groups[prompt_id] = low_quality
        if all_failed:
            n_all_failed_groups += 1
        elif low_quality:
            n_low_quality_groups += 1

        group_reward = strategy.score_group(
            GroupScores(scores=scoreable, all_failed=all_failed, low_quality=low_quality)
        )
        overall_rewards = group_reward.overall

        for idx, _ in group["samples"]:
            if idx in language_filtered:
                rewards[idx] = {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
            else:
                rewards[idx] = {
                    "overall":  overall_rewards.get(idx, 0.0),
                    "format":   1.0,
                    "accuracy": round(raw_scores[idx] / 10.0, 4),
                }

    # ------------------------------------------------------------------ #
    # Step 4 — diagnostic log (mean should be ≈ 0.500)                   #
    # ------------------------------------------------------------------ #
    overall_values = [r["overall"] for r in rewards]
    expected_note = "(expected≈0.500)" if _REWARD_TYPE == "rank" else "(raw mode — no fixed expectation)"
    print(
        f"[creative_solver_caller] "
        f"groups={len(groups)}  samples={n_samples}  workers={max_workers}  "
        f"reward_type={_REWARD_TYPE}  language_filtered={len(language_filtered)}  "
        f"mean_overall_reward={mean(overall_values):.3f}  {expected_note}",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # Step 5 — W&B metrics                                                #
    # ------------------------------------------------------------------ #
    try:
        wb = _get_wandb()
        if wb is not None:
            domain_raw: dict = defaultdict(list)
            for group in groups.values():
                wp = group["wp"]
                for idx, _ in group["samples"]:
                    domain_raw[wp.domain_name].append(raw_scores[idx])

            n_groups = len(groups)
            log_dict = {
                "solver/mean_rank_reward":    mean(overall_values),
                "solver/mean_raw_score":      mean(raw_scores) if raw_scores else 0.0,
                "solver/num_groups":          n_groups,
                "solver/num_samples":         n_samples,
                "solver/num_all_failed_groups":  n_all_failed_groups,
                "solver/pct_all_failed_groups":  n_all_failed_groups / n_groups if n_groups else 0.0,
                "solver/num_low_quality_groups": n_low_quality_groups,
                "solver/pct_low_quality_groups": n_low_quality_groups / n_groups if n_groups else 0.0,
                "solver/num_language_filtered":  len(language_filtered),
                "solver/pct_language_filtered":  len(language_filtered) / n_samples if n_samples else 0.0,
                "solver/num_truncated":          sum(truncated_flags),
                "solver/pct_truncated":          sum(truncated_flags) / n_samples if n_samples else 0.0,
            }
            for domain_name, scores in domain_raw.items():
                key = domain_name.lower().replace(" ", "_")
                log_dict[f"solver/domain/{key}/mean_raw_score"] = mean(scores)

            reason_counts: dict = defaultdict(int)
            for reason in failure_reasons:
                reason_counts[reason] += 1
            for reason, count in reason_counts.items():
                log_dict[f"solver/failure_reason/{reason}"] = count

            wb.log(log_dict, step=_solver_step)
    except Exception as _wb_err:
        print(f"[creative_solver_caller] W&B logging failed: {_wb_err}", flush=True)

    # ------------------------------------------------------------------ #
    # Step 6 — per-rollout JSONL logging                                  #
    # ------------------------------------------------------------------ #
    try:
        _log_solver_rollouts(
            groups, raw_scores, failure_reasons, truncated_flags, rewards, low_quality_groups
        )
    except Exception as _log_err:
        print(f"[creative_solver_caller] rollout logging failed: {_log_err}", flush=True)

    return rewards
