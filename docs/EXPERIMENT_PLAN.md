# R-Zero for Creative Writing — Diagnosis, Research Direction, and 7-Week Plan

*Date: 2026-08-03 · Target: ICLR 2027 (abstract ~mid-Sept, paper ~late Sept 2026)*

---

## 1. Diagnosis: why rewards are zero-heavy

The observed stats (challenger: 77.3% zeros; solver: 79.45% accuracy exactly 0.0, mean rank reward 0.109 vs. expected 0.5) are **mostly pipeline artifacts, not quality signal**. Key evidence: the judge prompt forces integer scores 1–10, so any *successful* judge call yields raw_score ≥ 1 → accuracy ≥ 0.1. **An accuracy of exactly 0.0 can only come from a failure path** (judge exception after 3 retries, language filter, or empty/invalid rollout) — and that's 79.45% of solver rows. Similarly, mean rank reward 0.109 ≈ 0.5 × 22% implies ~78% of groups were fully zeroed.

### H1 — Wrong format prompt (CONFIRMED) + token-budget truncation

`creative_challenger_smoke.sh` and `creative_solver_smoke.sh` don't override `data.format_prompt`, so `examples/config.yaml`'s default **`math_format.jinja`** is appended to every creative prompt: *"think about the reasoning process as an internal monologue … final answer MUST BE put in `\boxed{}`"*.

**Confirmed by observation:** solver outputs echo the template text verbatim — e.g. *"(first think about the reasoning process as an internal monologue, then present the final answer)𪨏"* — and the stray CJK char then trips the language filter → reward 0. Classic base-model prompt-echo + degeneration under an incoherent instruction.

**Can config parse/strip `\boxed{}` or think blocks from outputs? No.**
- *Input side:* `verl/utils/dataset.py::_build_messages` dispatches on **sentinel strings in the jinja file's content** (`"questioner_format_with_persona"`, `"questioner_format"`, `"solver_format"` → hardcoded math system prompts ignore the template entirely); any other content is jinja-rendered and appended to the prompt. There is no creative branch. This hidden dispatch is why the default was easy to miss.
- *Output side:* `verl/workers/reward/function.py` detokenizes rollouts and passes the **raw** `response_str` to `compute_score`. No post-processing hook exists in config. Any stripping must live in the reward function — but the right fix is upstream: a `creative.jinja` (plain `{{ content | trim }}` or a short genre-appropriate instruction) so there is nothing to strip.

Secondary effect: challenger `max_response_length=2048` including any preamble the template induces → truncated `<output>` JSON → `format=0`. **Verify:** log `finish_reason` / response token counts; fraction of format failures that are truncations vs. genuinely malformed JSON.

### H2 — Thinking-trace handling (deferred — thinking not currently enabled)

`apply_chat_template_kwargs: {}` and a base model → no true thinking mode in training rollouts today. The pseudo-`<think>` text that appears is induced by `math_format.jinja` and disappears with H1's fix. Keep for the future: `creative_solver_caller.py` never calls `split_thinking` before judging or language-filtering, so this becomes a real bug the day thinking is enabled. Add the strip now as cheap hardening.

### H3 — Language-filter false positives

`is_english_output` regexes CJK over the **full rollout including thinking**. Qwen3 models frequently emit Chinese tokens in reasoning → valid English answers get zeroed. **Verify:** count `language_filtered` lines in stdout logs; re-run filter on answer-only text.

### H4 — `_MIN_SCORE` scale bug / dead gate (fix, but not currently causal)

`_MIN_SCORE = 0.3` is compared against `raw_score` on a **1–10 scale** (comment says "if all of them are 0.1" — written as if scores were 0–1). A successful judge call returns ≥ 1.0, so the gate only fires when the entire group failed to score — at which point it converts what would be uniform 0.5s into all-zeros. Not the driver of the current stats, but fix the scale (or replace with an explicit all-failed-group path) before it silently changes behavior when the intended 0–1 semantics get "corrected" elsewhere.

### H5 — Challenger "uncertainty" reward is a broken estimator (conceptual; this is the research opportunity)

R-Zero's reward `1 − 2|p̂ − 0.5|` assumes p̂ = empirical solve rate over *m* solver samples against a **majority-vote pseudo-label**. The creative adaptation substitutes p̂ = (judge score of **one** solver sample)/10. Consequences:

- Single-sample, integer-quantized, judge-noise-dominated estimate — enormous variance.
- No pseudo-label mechanism exists at all in the open-ended setting; the theoretical grounding is gone.
- **Inverted curriculum in the low-score regime:** base-model outputs score ~2–4/10, i.e., p̂ < 0.5 almost always. There, `1 − 2|p̂ − 0.5|` is *monotonically increasing* in score → the challenger is rewarded for generating **easier** prompts, the opposite of the intended difficulty targeting.
- No repetition/diversity penalty (the math pipeline's `caller_penalty.py` BLEU penalty was dropped) → challenger can collapse to one prompt family.

### H6 — Rank reward noise: ties and double normalization

Judge scores are averages of ~5 integers; within a group of G=4, ties are common. `sorted()` breaks ties arbitrarily → tied samples get rank rewards {1.0, 0.67, 0.33, 0.0} by chance — pure noise injected into GRPO (which then z-scores the already-rank-normalized values again). When within-group true quality spread < judge noise (±1 point), the ranking is mostly noise.

**H6 verification checklist:**

1. **Tie rate:** from existing reward logs, per group compute the number of distinct `raw_score` values; report % of groups with ≥2 tied samples and % of samples whose rank depends on tie-breaking.
2. **Rank stability (test-retest):** re-judge the same G responses for ~50 groups; compute within-group Kendall τ between the two rankings. τ ≪ 1 ⇒ rank reward is mostly judge noise.
3. **SNR estimate:** compare within-group std of `raw_score` to judge test-retest std on identical responses. Signal exists only where group spread > judge noise; report the fraction of groups above that bar.
4. **Tie-break sensitivity:** recompute rank rewards with reversed tie order; measure how many rewards change and the induced advantage flip rate — a direct lower bound on injected gradient noise.
5. After fixes, re-check `mean_rank_reward ≈ 0.5` and rank-reward histogram uniformity per group size.

### Verification checklist (do before any redesign)

1. Add a `failure_reason` field to reward logs: `{judge_exception, rate_limit, language_filter, truncated, empty_answer, ok}`. Re-run one rollout batch → quantify each bucket.
2. Judge reliability audit: score ~200 fixed responses twice → test-retest correlation; compare Claude judge vs. WritingBench's official judge protocol on the same responses.
3. Token-length histogram of rollouts; % hitting max_response_length.
4. Dump one fully rendered prompt per run (post-`format_prompt`, post-chat-template) into the run log — this single line would have caught H1.

---

## 2. Research direction and narrative

**Title-shaped claim:** *Self-evolving curricula beyond verifiable rewards: adapting challenger–solver co-evolution (R-Zero) to open-ended generation.*

R-Zero's loop depends on two things that only exist in verifiable domains: (1) majority-vote pseudo-labels giving a ground truth for free, and (2) a binary correctness signal making `p̂ = 0.5` a principled "maximum learnability" target (it maximizes Bernoulli variance, hence GRPO advantage spread). Creative writing has neither. The paper's contributions:

1. **Negative result / diagnosis (cheap, already half-done):** naive transfer — judge-score-of-one-sample as p̂, rank rewards over noisy judge scores — produces degenerate signal. Characterize why: estimator variance, judge noise floor vs. within-group quality spread (SNR analysis), inverted curriculum in the low-score regime.
2. **Method:** principled replacements for both rewards.
   - **Solver reward:** noise-aware group ranking — pairwise-tournament or score z-scoring with tie handling; judged on thinking-stripped answers; optionally judge-ensembled.
   - **Challenger reward:** replace "uncertainty" with **learnability = within-group dispersion of solver scores across G samples** (the continuous analog of Bernoulli variance at p=0.5: prompts where solver sometimes succeeds, sometimes fails ⇒ maximum GRPO signal), plus a diversity/repetition penalty and format gate.
3. **Empirical:** WritingBench gains over (a) base model, (b) untrained-challenger prompts + solver RL (the "Base Challenger" analog), (c) static-dataset RL (RL on fixed WritingBench-style prompts — isolates the co-evolution contribution). Iteration-scaling curve (does iter 2–3 keep helping, as in R-Zero? Their own iteration-scaling caveat suggests a good analysis section). Reward-hacking audit (length bias, purple prose).

This is a strong story regardless of outcome magnitude: even modest gains + a rigorous account of *what breaks and why* when self-play RL meets non-verifiable rewards is an ICLR-viable contribution; the diagnosis section is publishable substance, not overhead.

---

## 3. 7-week experiment plan (Aug 3 → ~Sept 21)

**Week 1 — Fix ALL pipeline problems + refactor foundation (see `docs/REFACTOR_PLAN.md`).**

Fixes (each with a verification step):
1. `creative.jinja` format prompt; set `data.format_prompt` in both creative training invocations. Verify: dump one rendered prompt per run; no `\boxed{}`/monologue text in rollouts.
2. Reconcile env-var contracts: `modal_run.py` exports `RUN_ID` / `SOLVER_MAX_STEPS` / `CHALLENGER_MAX_STEPS` while the scripts read `SMOKE_RUN_ID` / `SMOKE_SOLVER_MAX_STEPS` / `SMOKE_CHALLENGER_MAX_STEPS`. Actual runs did use G=8, so the values reach the scripts through some launch path — but the names in these two files don't line up as written; trace how they actually flow, unify the names, and add validation. Preferably eliminated entirely by the refactor (typed args, no env hand-off).
3. `failure_reason` taxonomy in both reward callers + judge backoff on rate limits; language filter counts logged.
4. Fix `_MIN_SCORE` scale; tie-aware (average-rank) ranking.
5. Token budgets: challenger response 2048 → 4096; log truncation rate (`finish_reason`).
6. Cheap hardening: `split_thinking` before judge + language filter in solver caller (no-op today, correct tomorrow).
7. Refactor phase 1–2 (config unification + reward module extraction) so weeks 3–4 ablations are config-only.
- Exit criteria: <5% rollouts zero-due-to-failure; challenger format-valid rate >80%; `mean_rank_reward ≈ 0.5`; rerun of the distribution stats becomes the paper's "naive transfer" figure.

**Week 2 — Judge validation + baselines.**
- Judge audit (test-retest, vs. WritingBench official protocol, Haiku-vs-Sonnet agreement). Pick the training judge; keep a stronger held-out eval judge.
- Run WritingBench eval: Qwen3-4B-Base zero-shot (the floor), and existing checkpoints if any.
- Rerun one solver + one challenger rollout batch; regenerate the distribution stats → this becomes the paper's "naive transfer" diagnostic figure.

**Week 3 — Solver-side reward ablation (fixed, untrained challenger prompts).**
- Arms: (a) raw judge score, (b) current normalized rank, (c) tie-aware rank / z-score, (d) pairwise tournament. Short runs, judge-cost-matched.
- Select by WritingBench delta + reward SNR (within-group score variance vs. judge test-retest noise).
- This yields the "Base Challenger" baseline row for free.

**Week 4 — Challenger-side reward ablation.**
- Arms: (a) single-sample uncertainty (current, as the strawman), (b) within-group score variance / learnability over G solver samples, (c) target-band difficulty (reward mean score in [0.35, 0.65]); all + repetition penalty.
- Evaluate challenger *directly*: prompt diversity (self-BLEU, embedding dispersion), difficulty distribution against the frozen solver, domain coverage vs. WritingBench taxonomy.

**Week 5 — Full co-evolution, main results.**
- 2–3 iterations with the winning reward pair. Main table: Base / Base-Challenger / static-dataset RL / iters 1–3 on WritingBench (per-domain + overall).
- Start reward-hacking audit: length-controlled scoring, qualitative samples, held-out judge.

**Week 6 — Ablations, analysis, robustness.**
- Ablate: G (rollout n), diversity penalty, judge choice, learnability vs. uncertainty reward.
- Seeds: ≥2–3 for the headline number. Transfer check on a second creative eval (e.g., held-out WritingBench domains or another creative benchmark) to argue generalization.
- Freeze experiments end of week except reruns.

**Week 7 — Writing + buffer.**
- Paper, figures (reward-distribution before/after fix, SNR analysis, iteration scaling), reproduce-from-log checks. Buffer for one rerun.

**Standing risks:** judge API cost/rate limits (mitigate: Haiku for training reward, cache, batch API); Modal GPU budget (smoke-scale ablations, full scale only for winners); co-evolution instability at iter 2+ (known R-Zero limitation — have the iteration-scaling analysis ready as a finding, not a failure).

---

## 4. Immediate code changes (file-level)

| File | Change |
|---|---|
| `scripts/creative_{solver,challenger}_smoke.sh` | Set `data.format_prompt` to a new `creative.jinja`; revisit `max_response_length` (challenger ≥4k if thinking kept; solver 8k+ or disable thinking) |
| `examples/reward_function/creative_solver_caller.py` | `split_thinking` before judging and language filter; tie-aware ranks; fix/remove `_MIN_SCORE`; add `failure_reason` logging |
| `examples/reward_function/creative_writing_caller.py` | Replace single-sample uncertainty with G-sample dispersion reward; add repetition penalty (port from `caller_penalty.py`); log truncation |
| `evaluation/writing_bench/batch_eval_agent.py` | Backoff on rate limits; return failure reason instead of bare ValueError |
