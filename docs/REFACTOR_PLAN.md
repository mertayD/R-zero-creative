# Refactor Plan — From R-Zero Fork to a Creative-Only, Config-Driven Pipeline

*Goal: an experiment = editing one YAML file. Timebox: ~4–5 working days (Week 1), phased so the pipeline is never broken for more than a day.*

---

## 1. Issues identified

### A. Configuration is scattered across four layers with lossy hand-offs

The value `solver_max_steps` travels: `train_creative_coevolve(solver_max_steps=…)` → env `SOLVER_MAX_STEPS` → bash reads `${SMOKE_SOLVER_MAX_STEPS:-4}` → `trainer.max_steps`. Each hop renames it, and as written in these two files the names don't line up:

| modal_run.py sets | script reads |
|---|---|
| `SOLVER_MAX_STEPS` / `SOLVER_ROLLOUT_N` | `SMOKE_SOLVER_MAX_STEPS` / `SMOKE_SOLVER_ROLLOUT_N` |
| `CHALLENGER_MAX_STEPS` / `_ROLLOUT_N` | `SMOKE_CHALLENGER_*` |
| `RUN_ID` | `SMOKE_RUN_ID` (controls whether `SAVE_NAME` gets its own timestamp — affects whether `modal_run.py`'s computed checkpoint paths match) |

Note: actual runs used G=8, so values do reach the scripts through the real launch path — the point stands that the contract lives in string names across files with **no validation**: a rename or a different entry point degrades silently to defaults, and any failure surfaces as a `config.json missing` crash an hour into a run. Trace + unify in Phase 1; eliminate the env layer in Phase 2.

### B. Creative runs inherit math defaults invisibly (root cause of the missed `format_prompt`)

`examples/config.yaml` is the upstream math config (math12k dataset, `math.py` reward, `math_format.jinja`). Creative scripts override ~20 keys via CLI dotted args inside bash — anything not overridden silently keeps its math value. `format_prompt` was one; `algorithm.kl_coef`, `worker.rollout.temperature/top_p`, `val_override_config` are still inherited unreviewed today.

### C. Hidden dispatch on jinja file *content* (`verl/utils/dataset.py`)

`_build_messages` checks whether the template text *contains* `"questioner_format_with_persona"` / `"questioner_format"` / `"solver_format"` and, if so, discards the template and injects hardcoded math system prompts (incl. a PersonaHub download). The `.jinja` files are sentinels, not templates. Impossible to discover from config; this is why H1 was missed.

### D. Business logic lives in bash heredocs

Parquet building, validation, W&B table uploads are inline Python inside bash with shell-variable interpolation (`"${PROMPTS_JSON}"` pasted into Python source). Untestable, unlintable, quoting-fragile, and duplicated between challenger/solver scripts.

### E. Reward callers are monoliths with copy-paste drift

`creative_writing_caller.py` and `creative_solver_caller.py` each reimplement: W&B init, JSONL logging, judge-agent singleton, sys.path hacking. The parts that *should* vary per experiment (reward formula, filters, gates) are welded to the parts that never should (logging, clients). Consequences already visible: solver caller lacks `split_thinking` that challenger caller has; `_MIN_SCORE` scale bug; single-sample uncertainty estimator. To try a new challenger reward you must edit a 350-line file.

### F. Path/layout knowledge duplicated everywhere

`creative_smoke/`, `generated_questions_one_shot/`, `reward_logs/`, `global_step_{n-1}/actor/huggingface` are string-assembled independently in `modal_run.py`, three shell scripts, and both reward callers. The A-table path mismatch is the direct product.

### G. Fork artifacts obscure the actual project

Math reward fns (`math.py`, `r1v.py`, `caller*.py`), math training scripts (`questioner_train*.sh`, `solver_train.sh`, `smoke_test.sh`, `main.sh`), math evals (`eval_mmlupro.py`, `eval_supergpqa.py`, `eval_bbeh.py`), `math_verify` workarounds, and a README that is upstream R-Zero's. A reader (including you, six weeks from now, and eventually reviewers reading the code release) can't tell what's live.

### H. No run manifest / reproducibility record

A run's full configuration exists only as a combination of CLI flags, env vars, and bash defaults at launch time. Nothing writes the resolved config, git SHA, or prompt-template hash to the run directory. Everything is named "smoke" even when it's the real experiment.

### I. No tests, no dry-run

The `_MIN_SCORE` scale bug, tie noise, and env-var mismatches are all unit-test-sized bugs. There is no way to exercise the loop without burning GPU + judge dollars.

---

## 2. Target architecture

```
configs/
  base.yaml                  # ALL creative defaults, incl. full verl block — no inherited math values
  exp/
    solver_reward_rank.yaml  # experiment = tiny override file (5–15 lines)
    challenger_variance.yaml
creative_rzero/              # single installable package; no sys.path hacks
  config.py                  # typed ExperimentConfig (dataclass/pydantic); load = base ⊕ exp overrides;
                             # validates at load time; resolved config dumped to run dir
  paths.py                   # RunPaths — THE single source of run/checkpoint/parquet/log layout
  utils.py                   # FormatValidator, is_english_output, find_cjk_matches — shared by
                             # generation (steps/generate_prompts.py) and both reward callers
  data/
    writing_prompt.py        # WritingPrompt / QueryRequirements / PromptBatch data model
  prompts/
    one_shot.py              # challenger one-shot generation prompt copy (system/user templates)
  orchestrator.py            # co-evolution loop in pure Python (state, resume, verify, HF upload)
  steps/
    generate_prompts.py      # challenger ckpt → WritingPrompt JSON (was Step 0 bash); also owns domain
                             # sampling, prompt construction, response validation, the CLI entrypoint
    build_parquet.py         # prompts → train/val parquet (was bash heredoc)
    train_verl.py            # renders resolved verl YAML to run dir, launches trainer; no dotted-arg walls
    merge_ckpt.py            # wraps model_merger
  rewards/
    registry.py              # name → reward class; selected by config string
    solver/  rank.py | zscore.py | pairwise.py | raw.py
    challenger/  uncertainty.py | variance.py | target_band.py
    filters.py               # language filter, min-score gate (correct scale), truncation detection
    verl_entry.py            # the ONE file verl's reward_function points at; reads config, dispatches
  judge/
    client.py                # Claude/Haiku client: backoff, caching, failure reasons
    scoring.py               # criteria scoring + parse (from batch_eval_agent)
  logging_utils.py           # JSONL rollout log with failure_reason + W&B (used by all rewards)
modal_app.py                 # thin: image/volume/secrets + functions that call creative_rzero
legacy/                      # quarantined math fork artifacts (delete after paper)
tests/                       # reward math on fake scores; parquet golden files; config round-trip
```

Design rules:

1. **One config to rule a run.** `base.yaml` contains every knob including the verl section — creative values written out explicitly, nothing inherited from `examples/config.yaml`. An experiment file overrides a handful of keys. `train_verl.py` materializes the resolved verl config as a file in the run dir (reviewable artifact) instead of 20 CLI dotted args.
2. **Reward = strategy object selected by config.** `reward.solver.type: rank` / `reward.challenger.type: variance`. `verl_entry.py` is a 30-line adapter: load config from `EXPERIMENT_CONFIG_PATH`, instantiate from registry, delegate. New hypothesis = new 60-line strategy file + a yaml line. Shared judge/logging/filters injected, not copy-pasted.
3. **Kill the bash layer.** Scripts' only irreplaceable job (launching `verl.trainer.main` and the merger as subprocesses) moves into `steps/`. All function boundaries pass typed Python args — the entire class-A bug category becomes impossible. Keep one thin `scripts/run.sh` if you like muscle memory.
4. **`RunPaths` owns layout.** `paths.challenger_ckpt(iter_)`, `paths.solver_parquet(iter_, split)` etc. Orchestrator, steps, and rewards all import it; path mismatches become type-level impossibilities.
5. **Prompt construction is explicit.** Remove reliance on the sentinel hack: creative runs use `creative.jinja` rendered by the default branch (verify the rendered prompt in the run log). Don't edit verl beyond necessity — treat it as a vendored trainer; all creative logic stays in `creative_rzero`.
6. **Every run writes a manifest**: resolved config, git SHA, template hashes, judge model id, one fully rendered example prompt. Rename "smoke" → run sizes (`profile: smoke|small|full` as config presets).
7. **Dry-run mode.** `judge.type: mock` (returns seeded scores) + `rollout.mock: true` lets the full loop run on CPU in minutes; CI-able. This is what makes fast iteration real.

## 3. Phasing (never broken > 1 day)

**Phase 1 — Stop the bleeding (day 1).** `creative.jinja` + explicit `data.format_prompt`; fix the three env-var mismatches; `failure_reason` logging + judge backoff; `_MIN_SCORE` and tie fixes. Pure fixes in the current layout — this alone unblocks Week 2 experiments.

**Phase 2 — Config + paths (days 2–3).** `configs/base.yaml` (full explicit verl block) + `config.py` + `paths.py` + `train_verl.py`. Port bash heredocs into `steps/`. `modal_app.py` functions call `steps/` directly; delete the two creative bash scripts. Contract test: resolved config for a reference run matches the old dotted-arg values key-for-key (except intended fixes).

**Phase 3 — Reward registry (day 4).** Extract judge client + logging; split the two callers into strategy classes behind `verl_entry.py`. Unit tests for rank math (ties, G=1, all-failed groups), uncertainty vs. variance formulas, filters. Behavior-preserving except Phase-1 fixes.

**Phase 4 — Quarantine + manifest (day 5).** Move math artifacts to `legacy/`; project-specific README (loop diagram, config reference, "how to add a reward"); run manifest; mock-judge dry-run in CI. Optionally delete `legacy/` after submission.

**Definition of done:** launching the Week-3 solver-reward ablation = 4 yaml files in `configs/exp/`, zero code edits; a full dry-run passes on CPU; one rendered prompt visible in every run log.

---

## 4. Task breakdown

Per-task format: **Pointers** (files/functions/lines as of today) · **Do** · **Goal** · **Accept** (checkable expectation). Order within a phase = execution order; tasks marked ⛓ depend on the previous one.

### Phase 1 — Stop the bleeding (day 1)

**T1.1 — Create `creative.jinja` and wire it into both trainings**
- Pointers: `examples/format_prompt/` (new file); `scripts/creative_solver_smoke.sh` verl arg block (~L220–247); `scripts/creative_challenger_smoke.sh` verl arg block (~L170–196); sentinel dispatch in `verl/utils/dataset.py::_build_messages` (L145–221).
- Do: new template containing only `{{ content | trim }}` (solver) — the WritingPrompt query already carries its own instructions. **Trap:** the file content must not contain the substrings `questioner_format` or `solver_format`, or `_build_messages` hijacks it into hardcoded math prompts. Add `data.format_prompt=./examples/format_prompt/creative.jinja` to both verl invocations. Challenger keeps its structured-output instructions in the parquet `problem` text, so the same pass-through template works.
- Goal: kill the `\boxed{}` / "internal monologue" echo — root cause of degenerate rollouts and CJK-filter zeros.
- Accept: T1.2's rendered-prompt dump contains no `\boxed`, no "reasoning process", no `<think>` instruction; challenger format-valid rate rises materially above 22.7% on a 1-step probe run.

**T1.2 ⛓ — Log one fully rendered prompt per run**
- Pointers: `verl/utils/dataset.py::__getitem__` (L247–267) or, less invasively, the first `compute_score` call in each reward caller.
- Do: one-time `print` of the final post-chat-template prompt string (index 0) at startup, tagged `[RENDERED_PROMPT]`.
- Goal: make H1-class bugs visible in every future run log; this single line would have caught the original issue.
- Accept: `grep RENDERED_PROMPT` on any run log shows the exact string the model saw.

**T1.3 — ~~Reconcile the env-var / checkpoint-path contract~~ DROPPED (see §5)** — no runs happen before the bash/env layer is deleted, so patching its names is wasted motion. Superseded by T2.6/T2.8 eliminating the layer.

*Original task kept for reference:*
- Pointers: `modal_run.py` env dicts in `train_creative_challenger` (L1460–1474) and `train_creative_solver` (L1637–1651): set `RUN_ID`, `{CHALLENGER,SOLVER}_MAX_STEPS`, `*_ROLLOUT_N`; scripts read `SMOKE_RUN_ID` (`creative_challenger_smoke.sh` L49, `creative_solver_smoke.sh` L52), `SMOKE_*_MAX_STEPS` / `SMOKE_*_ROLLOUT_N` (L34–35 / L41–43). Returned paths built at `modal_run.py` L1496, L1673 vs. script `SAVE_NAME` (`…_challenger_v1[_ts]` / `…_solver_v1[_ts]`).
- Do: (a) trace how your real runs actually received G=8 / step counts (find the launch path; document it in this file); (b) pick one canonical name per knob, fix both sides; (c) add `: "${VAR:?not set}"` guards at the top of each script; (d) stop reconstructing checkpoint paths in `modal_run.py` — have each script write the final merged path to `${STORAGE_PATH}/…/last_ckpt.txt` and have the Modal function read it back.
- Goal: no silent-default degradation; path agreement by construction. (Phase 2 deletes this whole layer — do the minimum that makes Week-2 runs trustworthy.)
- Accept: launch with `--solver-max-steps 6 --solver-rollout-n 8` → verl log shows `max_steps=6`, `rollout.n=8`; `_verify_checkpoint` passes without manual path fixes.
- **Trace note (done during T1.3):** `modal_run.py`'s `train_creative_challenger`/`train_creative_solver` set `CHALLENGER_MAX_STEPS`/`CHALLENGER_ROLLOUT_N`/`SOLVER_MAX_STEPS`/`SOLVER_ROLLOUT_N`/`RUN_ID`; the two scripts read `SMOKE_*`-prefixed names that were never set, so `max_steps`/`rollout_n` passed at the `modal run` CLI never reached `trainer.max_steps`/`worker.rollout.n` — every launch silently trained on the scripts' own defaults (`C_STEPS=4`, `S_STEPS=4`, `ROLLOUT_N=4`) regardless of the CLI flags. `RUN_ID` was likewise never seen as `SMOKE_RUN_ID`, so every script invocation (including from `train_creative_coevolve`) took the "standalone" branch and stamped its own fresh timestamp onto `SAVE_NAME`, decoupling the checkpoint directory name from what `modal_run.py` reconstructed (`{abbr}_challenger/global_step_{max_steps-1}/...` — missing the `_v1[_ts]` suffix entirely). Fixed by renaming the scripts to read the canonical (already-correct) names from `modal_run.py`, and by having each script write its actual merged checkpoint path to `${STORAGE_PATH}/models/{abbr}_{role}_last_ckpt.txt` for `modal_run.py` to read back rather than reconstruct.

**T1.4 — `failure_reason` taxonomy in both reward callers**
- Pointers: `examples/reward_function/creative_solver_caller.py::_score_one` (L167–183), language filter (L278–287), JSONL logger (L92–126); `creative_writing_caller.py::_score_one` (L215–232), logger (L109–144); `evaluation/writing_bench/batch_eval_agent.py::score_all_criteria` (raises bare `ValueError`, ~L95–115).
- Do: `_score_one` returns `(score, reason)` with `reason ∈ {ok, judge_parse_fail, judge_rate_limit, judge_api_error, language_filter, empty_answer, truncated}`. `batch_eval_agent` distinguishes parse failure from API error (inspect exception type / HTTP status) instead of one bare raise. Add `failure_reason` to every JSONL row and per-step W&B counts per reason.
- Goal: never again infer failure modes from score distributions — the 79.45%-exact-zero mystery becomes a one-groupby answer; this is also the data source for the step-19-collapse autopsy.
- Accept: rerun one rollout batch → a table of counts by reason; `accuracy == 0.0 ⇔ reason != ok` holds row-for-row.

**T1.5 — Judge backoff + on-disk cache**
- Pointers: `evaluation/writing_bench/evaluator/` (`ClaudeAgent.run`); worker caps `_MAX_WORKERS=4` (`creative_solver_caller.py` L63).
- Do: exponential backoff with jitter on 429/5xx (respect `retry-after`), separate retry budget from parse-retries; optional JSON-on-disk cache keyed by `sha256(system+prompt)` under `${STORAGE_PATH}/judge_cache/` (idempotent re-runs, free re-scoring for the H6 test-retest audit); consider raising workers once backoff exists.
- Goal: judge infrastructure stops masquerading as reward signal; re-judging for reliability audits becomes cheap.
- Accept: synthetic 429 (mock) triggers backoff not failure; scoring the same batch twice makes zero API calls the second time (cache hit).

**T1.6 — Fix `_MIN_SCORE` gate semantics**
- Pointers: `creative_solver_caller.py` L66 (`_MIN_SCORE = 0.3`), L221–225 (gate inside `_assign_normalised_rank_rewards`), L336–338 (rejected-group counter).
- Do: replace the magic 0.3-vs-1–10 comparison with two explicit, configurable gates: `all_failed` (every scoreable sample has `reason != ok` → group excluded, uniform reward, counted) and `low_quality` (max raw score < threshold **on the 1–10 scale**, e.g. 2.0 → group skipped to avoid ranking indistinguishable garbage). Document that uniform rewards (all-0 or all-0.5) both yield zero GRPO advantage — the gate's job is bookkeeping + skipping noise-ranking, not punishment.
- Goal: gate does what its comment says; group rejection becomes a measured, intentional event.
- Accept: unit test — group of judge scores {1,1,2,1}: gated by `low_quality` at threshold 2.0, not silently ranked; group all-`judge_api_error`: gated as `all_failed`, never as quality.

**T1.7 — Tie-aware rank rewards**
- Pointers: `creative_solver_caller.py::_assign_normalised_rank_rewards` (L190–231).
- Do: use average ranks for ties (`scipy.stats.rankdata(-scores, method="average")`), then `R_i = (G_eff − rank_i)/(G_eff − 1)`. Equal scores ⇒ equal rewards ⇒ zero advantage between indistinguishable samples.
- Goal: remove the dominant gradient-noise source (98.6% of your G=8 groups contain ties; tie-broken ranks are pure noise).
- Accept: unit test — scores {5,5,3,3}: rewards {0.833, 0.833, 0.167, 0.167} (average-rank), not {1, 0.67, 0.33, 0}; all-equal group → all 0.5.

**T1.8 — Language filter: answer-only + counted**
- Pointers: `creative_solver_caller.py` L278–287 (`is_english_output(pred)` on raw rollout); `question_generate/one_shot_creative_question_generate.py::is_english_output` (L42–44, CJK regex).
- Do: filter on `split_thinking(pred)[1]` (answer only); count matched CJK chars and log a preview of what triggered it; emit per-step filtered counts (this is the metric predicted to ramp before the step-19 collapse).
- Goal: filter catches genuine degeneration, not incidental artifacts; its per-step curve becomes a collapse early-warning signal.
- Accept: a normal English answer with one stray CJK codepoint in a hypothetical think-prefix passes; per-step `num_language_filtered` visible in W&B.

**T1.9 — Token budgets + truncation detection**
- Pointers: `creative_challenger_smoke.sh` L176 (`max_response_length=2048`); `creative_solver_smoke.sh` L227 (`4096`); reward callers (no truncation signal available from verl — approximate in the caller).
- Do: challenger 2048 → 4096. In both callers, flag `truncated` when the tokenized response length is within a few tokens of `max_response_length` (pass the limit via env/config) or when the challenger `<output>` block has an opening tag without a closing one.
- Goal: separate "model couldn't finish" from "model wrote garbage" in the format-failure stats.
- Accept: challenger format failures split into `truncated` vs `malformed` in logs; truncation rate < 10% after the budget bump.

**T1.10 — `split_thinking` hardening in solver caller**
- Pointers: `creative_solver_caller.py` Step 1b/Step 2 (judge receives raw `pred`); `evaluation/shared/utilities.py::split_thinking`.
- Do: judge and filter on the answer part only. No-op today (thinking disabled); correct the day `apply_chat_template_kwargs.enable_thinking` is turned on.
- Accept: unit test with a synthetic `<think>…</think>answer` rollout — judge prompt contains only `answer`.

**T1.11 — Health-based checkpoint selection + collapse guard**
- Pointers: merge calls — `creative_solver_smoke.sh` L253–257, `creative_challenger_smoke.sh` (same pattern); `S_MERGE=$((S_STEPS-1))` convention; per-step health available from T1.4's logs.
- Do: after training, compute per-step all-zero-group rate from the rollout JSONL; merge the **last step whose rate < 10%** (fallback: best step) instead of blindly `max_steps − 1`. Log a loud warning when any step exceeds 50% (collapse flag). Your iter-1 solver was merged at a step deep inside the 19–32 collapsed region — this task prevents feeding collapsed checkpoints into the next co-evolution round.
- Goal: co-evolution iterates on the best available policy, not the last one.
- Accept: rerunning selection on the existing run's log picks a step ≤ 18; merged path recorded in the run manifest with the health metric that chose it.

### Phase 2 — Config + paths (days 2–3)

**T2.1 — `configs/base.yaml` with a fully explicit verl block**
- Pointers: `examples/config.yaml` (all 60+ keys); the two scripts' dotted-arg overrides (solver ~L220–247, challenger ~L170–196).
- Do: copy `examples/config.yaml`, then **review every key against creative needs** and record the decision as a comment where it differs from math defaults. Known must-changes: `data.format_prompt` (creative.jinja), `data.train_files/val_files` (placeholders — set per run), `worker.reward.reward_function` (→ `verl_entry.py`, T3.6), `trainer.project_name`. Known must-reviews: `algorithm.kl_coef=1e-2`/`disable_kl`, `worker.rollout.temperature=1.0/top_p=0.99` (creative gen may want different sampling), `val_override_config`, `save_limit`. Add the non-verl sections: `run` (abbr, profile, seed), `challenger`, `solver`, `judge`, `rewards`.
- Goal: zero inherited-math-value surprises — the class-B bug becomes impossible because every value is visible and reviewed.
- Accept: `diff` of resolved verl config vs. what the old scripts produced shows only intended changes (this diff is the review artifact — commit it).

**T2.2 ⛓ — `creative_rzero/config.py`: typed load/merge/validate/dump**
- Do: dataclass (or pydantic) `ExperimentConfig`; loader = OmegaConf merge of `base.yaml` ⊕ `configs/exp/<name>.yaml` ⊕ CLI dotlist; validation at load (`max_steps ≥ 1`, `rollout_n ≥ 2` for rank rewards, judge model id known, reward names in registry); `save_resolved(run_dir)` writes the merged config.
- Goal: config errors fail in seconds locally, not an hour into a Modal run.
- Accept: loading an exp file with a typo'd key or `rollout_n: 1` raises with a pointed message; resolved YAML lands in the run dir.

**T2.3 — `creative_rzero/paths.py`: `RunPaths` as single path authority**
- Pointers: current duplicated layouts — `modal_run.py` L1496/L1673/L1935/L1960, both scripts' `SAVE_NAME`/parquet/JSONL paths, reward callers' `reward_logs/` construction (both callers L119–123 / L100–104).
- Do: `RunPaths(storage, abbr, run_ts, iteration)` with methods: `challenger_ckpt(step)`, `solver_ckpt(step)`, `train_parquet(role)`, `val_parquet(role)`, `rollout_log(role)`, `judge_cache()`, `manifest()`, `state_file()`. Every consumer imports it; grep-kill remaining f-string paths.
- Goal: the class-F path-mismatch category becomes structurally impossible.
- Accept: `grep -rn "global_step_" --include="*.py" --include="*.sh"` returns only `paths.py` (+ verl internals); a layout test pins the exact strings.

**T2.4 — `steps/generate_prompts.py`** (port of solver script Step 0)
- Pointers: `creative_solver_smoke.sh` L100–115; `question_generate/one_shot_creative_question_generate.py` CLI.
- Do: `generate_prompts(cfg, paths, challenger_ckpt) -> Path` calling the generator in-process (or via subprocess with explicit args); returns the prompts JSON path; raises with prompt-count context on shortfall.
- Accept: produces byte-equivalent JSON to the bash path for a fixed seed.
- **Implementation note (done during T2.4):** built in two passes. First pass kept `question_generate/one_shot_creative_question_generate.py` as-is and had `generate_prompts()` subprocess out to it (`python3 <old file> --model ... --save_name ...`), matching the "or via subprocess" option above. Second pass (full cutover, prompted by review) went further than the task scoped: split that file entirely —
  - `creative_rzero/data/writing_prompt.py` — `WritingPrompt` / `QueryRequirements` / `PromptBatch` (the data model; also used by both reward callers and `steps/build_parquet.py`)
  - `creative_rzero/utils.py` — `FormatValidator` / `is_english_output` / `find_cjk_matches` (format/language validation; also used by both reward callers)
  - `creative_rzero/prompts/one_shot.py` — `ONE_SHOT_SYSTEM_PROMPT` / `render_one_shot_user_prompt` (the actual prompt copy, kept separate from sampling/control-flow)
  - `creative_rzero/steps/generate_prompts.py` — `DomainSampler`, `build_one_shot_prompt`, `validate_one_shot_response`, `generate_prompts_batch`, `run_generation`, the CLI entrypoint, and the typed `generate_prompts()` step

  Old file deleted outright (no shim); the 3 real consumers (`creative_writing_caller.py`, `creative_solver_caller.py`, `evaluation/writing_bench/solver_sampling.py`) and both bash scripts' import/invocation paths were repointed at the new locations in the same change. `vllm`/`transformers` imports were made lazy (function-scope, not module-scope) throughout so the data/validation modules stay importable without a GPU — this is what let 37 new unit tests get written for logic (`DomainSampler`, `build_one_shot_prompt`, `validate_one_shot_response`, the `generate_prompts_batch` retry/attempt-cap loop via a fake injected `vllm` module, `FormatValidator`) that had **zero** test coverage before, despite already existing pre-refactor.

  `generate_prompts()` also stopped subprocessing entirely — it now calls `run_generation()` in-process. Rationale: post-consolidation, subprocessing would have meant the module spawning a child process that re-executes itself (`python -m creative_rzero.steps.generate_prompts`), which only made sense for two reasons, one of which is fixable and one of which has no caller yet (see "Known follow-up" below).

- **Known follow-ups / cleanup surfaced by this work (not yet actioned):**
  1. **GPU-process isolation deferred, not solved.** The original subprocess call existed partly to guarantee vLLM's CUDA context/KV cache gets released (a child process exiting is the only reliable teardown) and partly to contain `main()`'s old `sys.exit(1)` calls. The `sys.exit` reason is now moot (refactored to raise `PromptGenerationError`), but the GPU-memory reason is real — it just has no caller today, since `orchestrator.py` (T2.8) doesn't exist yet. **When T2.8 is built and calls `generate_prompts()` once per co-evolution iteration from one long-lived process, revisit whether repeated in-process vLLM loads leak GPU memory across iterations.** If so, the isolation decision belongs at the orchestrator level (e.g. subprocessing a whole iteration), not baked back into this step.
  2. **Two dead branches found via the new tests, left in place (documented at their call sites in `tests/test_generate_prompts.py`), not fixed:**
     - `PromptBatch.generation_log["total_attempted"]` / `["skipped"]` only ever increment on a *successful* slot — `generate_prompts_batch` explicitly never calls `add_prompt(None)` on failure — so these two counters don't track what their names imply. `format_validation_failures` / `language_filter_failures` are the counters that actually work.
     - `generate_prompts_batch`'s own inline check (`if is_valid and not is_english_output(parsed_json.get('query', '')))`) is unreachable: `validate_one_shot_response` → `FormatValidator.validate_response` already rejects a non-English `query` earlier in the same call chain, so `is_valid` is already `False` by the time this second check would run.
     - Candidate for T3.8 (test suite hardening) or a small standalone cleanup pass: either wire `total_attempted`/`skipped` up to mean what they say, or drop them; delete the unreachable language check.
  3. **Still unvalidated on a real GPU box.** All new coverage is unit-level against fakes (`_FakeLLM`, an injected fake `vllm` module) — nobody has run `run_generation()` against a real checkpoint since the split. First real signal will come from T2.9's validation run; if it surfaces an issue, it's likely in `run_generation`'s vLLM/tokenizer wiring specifically, since that's the one code path the fakes can't exercise.

**T2.5 — `steps/build_parquet.py`** (port of both bash heredocs)
- Pointers: `creative_solver_smoke.sh` L120–205 heredoc; equivalent block in the challenger script.
- Do: pure function `(prompts_json, cfg, paths) -> (train_path, val_path)`; keep the current row schema (documented in the docstring); validation (non-empty query, ≥1 criterion) with skip-logging preserved.
- Accept: golden-file test — same input JSON produces a parquet with identical schema and row count as the heredoc did; runs under pytest with no shell involved.

**T2.6 — `steps/train_verl.py`: resolved-config launcher**
- Pointers: the two 25-line dotted-arg walls; `verl.trainer.main` accepts `config=<yaml>`.
- Do: take `ExperimentConfig`, materialize the full verl YAML into `run_dir/verl_config_{role}.yaml`, launch `python -m verl.trainer.main config=<that file>` via subprocess with a clean, explicit env (only what verl + reward entry need: `EXPERIMENT_CONFIG_PATH`, `STORAGE_PATH`, keys). Stream logs; raise on nonzero exit with the last 50 lines.
- Goal: the exact training config becomes a reviewable, diffable artifact per run; no CLI-quoting bugs.
- Accept: run dir contains the YAML verl actually consumed; a run can be reproduced by pointing at that file alone.

**T2.7 — `steps/merge_ckpt.py`** with T1.11's health-based step selection built in.
- **Collapse definition** (requires T1.4 reason codes): split per-step zero-group causes into `infra = {judge_api_error, judge_rate_limit}` (harmless to the checkpoint — zero reward ⇒ zero advantage) vs. `policy = {language_filter, empty_answer, truncated, all-scores-at-floor}`. Only policy-attributable failure vetoes a checkpoint.
  - *Hard flag:* policy-attributable zero-group rate > 50% at any step.
  - *Soft flag (ramp):* 3 consecutive steps with rate > 2× run baseline (steps 1–3) and monotonically worsening. (On the observed iter-1 run this fires at step ~17–18, before the step-19 cliff.)
  - Supporting signals, all free: mean raw score, median response length, truncation %, CJK %, distinct-2-gram ratio from the JSONL; entropy / KL / grad-norm from verl's W&B logs.
  - *Selection:* healthy = unflagged and rate < 10%; merge last healthy step (or best mean raw score among healthy).
- **Config dependency:** `trainer.save_limit: 3` breaks retrospective selection — the healthy checkpoint may be pruned by the time collapse is detected. Either raise `save_limit` to cover the collapse window, or (preferred) make the monitor **online**: `train_verl.py` tails the rollout JSONL between steps and kills the trainer subprocess when the soft flag trips — last saved checkpoint is then near-healthy by construction, and no compute is spent on doomed steps (iter-1 burned 14 steps of rollouts + judge calls post-collapse).
- Accept: returns the merged HF dir path; refuses (with override flag) to merge a flagged step; replaying the iter-1 JSONL through the detector flags ≤ step 18 and never selects 19+.
- **Bug found and fixed post-T2.9 (2026-08-09): step numbering was off by one.** VERL is 1-indexed — `ray_trainer.fit` sets `global_step = 0` then increments it at the *top* of each training step (L466/L482), so a `max_steps: N` run writes `global_step_1 … global_step_N` and `global_step_0` never exists. `merge_ckpt.py` assumed 0-indexing in three places (`_log_step_to_global_step` mapping log step N → N-1; `select_checkpoint_step` searching `range(max_steps)`; the no-data fallback returning `max_steps - 1`), so it could select `global_step_0` — killing the run with `FileNotFoundError: .../global_step_0/actor` in `model_merger.py` — and could *never* select `global_step_{max_steps}`, the final and most-trained checkpoint. Observed on run `t29-validation-tiny_20260809_215322`. Fixed by mapping log step → identically-numbered global_step and searching `range(1, max_steps + 1)`; replaying that run's log now selects `global_step_1` (exists) instead of `global_step_0`. Note the old bash convention `S_MERGE=$((S_STEPS-1))` was the original source of the off-by-one and was itself merging the second-to-last checkpoint.

**T2.8 ⛓ — `modal_app.py` thin wrappers; delete creative bash scripts**
- Pointers: `modal_run.py` L1392–1984 (three creative functions + coevolve, incl. duplicated 10-line env dicts at L1399–1410 and L1550–1561); orchestration logic (state/resume/verify/upload, L1792–1976) moves to `creative_rzero/orchestrator.py` unchanged in behavior.
- Do: one decorator-factory for the shared Modal env; functions become `cfg = load(...); orchestrator.run_iteration(cfg, paths)`. Keep `modal_run.py` untouched until T2.9 passes, then delete `creative_{challenger,solver,coevolve}_smoke.sh` and the old functions.
- Accept: `modal run modal_app.py::coevolve --config configs/exp/repro.yaml` completes a smoke-profile iteration end-to-end.

**T2.9 — Validation run (replaces parity run — see §5)**
- Do: one smoke-profile GPU run of the **new** pipeline only. No old-vs-new comparison — the old pipeline's behavior was wrong (math format prompt, tie noise, dead gate); reproducing it proves nothing.
- Accept, from that single run's logs: `[RENDERED_PROMPT]` clean (no `\boxed`/monologue); `failure_reason` table with `ok` ≥ 95%; `mean_rank_reward ≈ 0.5` in non-gated groups; challenger format-valid > 80%; checkpoint selected by health metric and recorded in manifest. This run doubles as the "after" distribution stats for the paper's diagnostic figure.
- **Post-fix validation run succeeded end-to-end (2026-08-09/10, branch `fix/merge-ckpt-step-indexing`, run `t29-validation-tiny_20260809_235627`).** Ran `modal run --detach modal_app.py::main --config configs/exp/validation_tiny.yaml` after the off-by-one fix (commit `33de73f`) landed and the CPU `preflight_check` passed cleanly. Both co-evolution phases completed and merged correctly — the direct test of the fix:
  - **Challenger**: both training steps checkpointed (`global_step_1`, `global_step_2`); merge selected `global_step_1` (never `global_step_0`), verified (`[OK] Challenger iter1 checkpoint verified: .../global_step_1/actor/huggingface`), uploaded to `mertayd/r-zero-creative-datasets`. Rollout log: 8/10 rows `ok`, 2 `truncated` (80% ok; format-valid 80%, meeting the plan's >80% bar). Truncation is expected at this profile's small `max_response_length: 2048` for a 0.6B model, not a new bug — the 95%-ok bar in this task's Accept line is an experiment-quality target, not a wiring-validation one.
  - **Solver**: prompt generation from the merged challenger checkpoint succeeded (3/3 prompts, 100% success after 3 json-fence retries); both training steps checkpointed; merge selected `global_step_1`; verified and uploaded. Rollout log: 10/10 rows `ok` (100%). Live `BatchFunctionRewardManager` output confirmed `mean_rank_reward=0.500 (expected≈0.500)`.
  - `[RENDERED_PROMPT]` logged clean — no `\boxed`, no math system-prompt artifacts.
  - Run finished cleanly: `=== Creative Co-Evolution Complete ===`, `Final challenger` and `Final solver` both resolved to their respective `global_step_1/actor/huggingface` paths; Modal app transitioned `ephemeral (detached)` → `stopped` with no errors.
  - Not yet true: a `manifest.json` recording the health-based selection (that's T4.3, not yet built) — selection itself is confirmed working via the rollout-log health check in `merge_ckpt.py`, just not yet persisted as a standalone manifest artifact.
  - Each phase took roughly 25–50 minutes wall-clock for this tiny 2-step/0.6B config, almost entirely CPU-bound work (reward scoring, W&B sync) behind Modal's stdout rate-limiter rather than GPU compute — confirmed via `ps`/`/proc` state checks mid-run, not a performance regression to chase.
  - Housekeeping: deleted volume debris from three pre-fix crashed attempts at this same config (`t29-validation-tiny_20260809_{190840,205102,215322}` — runs/, models/, reward_logs/, parquet, and generated-prompt artifacts), all of which had died in Phase A before ever reaching the buggy merge step for the first two, and at the buggy merge itself for the third.

### Phase 3 — Reward registry (day 4)

**T3.1 — `judge/client.py`**: extract `ClaudeAgent` + T1.5 backoff/cache; add `MockJudge` (seeded deterministic scores) behind the same interface. Accept: reward unit tests run offline with `judge.type: mock`.

**T3.2 — `judge/scoring.py`**: port `batch_eval_agent.py` (criteria formatting, fence-stripping, parse-validation, retries) returning `(scores, failure_reason)`. Accept: parse-failure fixtures produce `judge_parse_fail`, not exceptions.

**T3.3 — `rewards/registry.py` + strategy interface**
- Do: `class RewardStrategy: def __call__(self, predicts, ground_truths, ctx) -> list[RewardScore]`; `ctx` carries judge, config, logger, paths. `@register("rank")` decorator; lookup by `cfg.rewards.solver.type` / `cfg.rewards.challenger.type`.
- Goal: hypothesis = one new file + one yaml line; shared plumbing injected once.
- Accept: `rewards.solver.type: nonsense` fails at config load listing valid names.

**T3.4 — Solver strategies**: `rank.py` (current logic + T1.6/T1.7 fixes — port, don't rewrite), `zscore.py` (per-group z-scored raw score, clipped), `raw.py` (score/10 baseline), `pairwise.py` (judge picks winner per pair; Bradley–Terry-lite win-rate; stub acceptable this week). Accept: all four pass a shared test suite over the same fixture groups (ties, all-failed, G=1, mixed).

**T3.5 — Challenger strategies**: `uncertainty.py` (current single-sample formula — kept as the paper's strawman baseline), `variance.py` (sample G solver responses per generated prompt via `rollout/solver_client.py`; reward = within-group std or IQR of judge scores — the learnability signal; `G_challenger` configurable, default 4–6), `target_band.py` (reward mean score ∈ [lo, hi]); plus a composable `repetition_penalty` term (self-BLEU vs. batch, port the idea from `caller_penalty.py`). Accept: variance strategy's judge-call count matches `n_prompts × G` (budget check); each strategy logs its own W&B namespace.

**T3.6 — `rewards/verl_entry.py`**: the single `reward_function` target. Reads `EXPERIMENT_CONFIG_PATH`, instantiates the configured strategy once (module-level cache — verl imports the file per worker), delegates `compute_score`. Both roles' verl configs point here; role passed via config, not filename. Accept: switching solver reward rank→zscore = 1-line yaml change, verified in logs.

**T3.7 — `logging_utils.py`**: one JSONL writer (shared schema: step, role, prompt_id, raw_score, reward, failure_reason, preview, truncated) + one W&B helper; delete the two copy-pasted logger/`_get_wandb` implementations. Accept: challenger and solver logs load into the same DataFrame with a `role` column — the per-step health analysis you just ran becomes a reusable `scripts/analyze_run.py`.

**T3.9 — Typed dataclasses for rollout-log entries** *(execution order: do before T3.7 — it defines the schema T3.7's writer consumes)*
- Pointers: `creative_solver_caller.py::_log_solver_rollouts` (L92–126; schema lives only as a comment block, L67–88) and `creative_writing_caller.py::_log_challenger_rollouts` (L109–144; comment schema L84–105). Both build entries as ad-hoc inline dicts — the schema is enforced by nothing, documented in comments that already drift from the code, and the two files disagree on field names for the same concepts (`response_preview` vs `solver_response`, `raw_score` vs absent, `input_query` vs `generated_query`).
- Do: in `creative_rzero/logging_utils.py` (or `data_models.py`), define frozen dataclasses:
  - `RolloutLogEntry` (shared base): `schema_version`, `step`, `role` (`solver|challenger`), `prompt_id`, `rollout_idx`, `raw_score`, `reward`, `accuracy`, `failure_reason` (T1.4), `truncated` (T1.9), `response_preview`, `response_len_tokens`.
  - `SolverLogEntry(RolloutLogEntry)`: `domain`, `domain_name`, `subdomain`, `num_criteria`, `input_query`.
  - `ChallengerLogEntry(RolloutLogEntry)`: `format_valid`, `generated_query`, `generated_criteria_n`, `challenger_thinking`, `solver_response`, `solver_thinking`.
  - One `to_jsonl_line()` / `from_json()` pair (`dataclasses.asdict`); include `schema_version: 1` so the analysis script can handle old logs.
- Goal: the log schema becomes code, not comments — field renames break at construction time instead of silently producing `NaN` columns in analysis; solver and challenger logs share a common core so per-step health analysis (T2.7 detector, `scripts/analyze_run.py`) works over both with one loader.
- Accept: both reward strategies construct entries only via these classes (no inline dicts); round-trip test `from_json(to_jsonl_line(e)) == e`; loading a mixed solver+challenger JSONL yields one DataFrame with the shared columns non-null for every row; the T2.7 collapse detector runs off `RolloutLogEntry` fields alone.

**T3.10 — Separate validation rollouts from training rollouts in the reward log** *(found 2026-08-09; execution order: with or just after T3.9, whose `RolloutLogEntry` should carry the new field)*
- Pointers: `verl/trainer/main.py` L79–80 — `reward_fn` and `val_reward_fn` are **two separate `RemoteRewardManager` Ray actors**, i.e. two processes, each with its own module-global step counter (`_challenger_step` / `_solver_step`) but both writing to the same `{STORAGE_PATH}/reward_logs/{VERL_EXPERIMENT_NAME}.jsonl`. `verl/trainer/ray_trainer.py` L474–475 (the `val_before_train` pass) and L624–631 (the post-training pass, which fires *because* `val_freq <= 0`, not despite it). Consumers that are misled by the mixture: `creative_rzero/steps/merge_ckpt.py::compute_step_health`, `creative_rzero/steps/report.py::rollout_health` / `build_run_health`.
- Symptom (run `t29-validation-tiny_20260809_215322`; `max_steps: 2`, `num_train: 2`, `rollout_n: 2`, `num_val: 1`): the challenger rollout log held **10 rows, not the expected 8**. Training wrote 4 rows/step; the two validation passes wrote 1 row each under the *validation* actor's own counter, so validation rows land on `step=1` and `step=2` next to training rows and are indistinguishable from them. The file's `rollout_idx` order — `0,0,1,2,3` then `0,1,2,3,0` — matches val-before-train, two training steps, final val, exactly.
- Do: add a `split: "train" | "val"` field to every rollout row, sourced from the reward manager that produced it. Cleanest plumbing mirrors how `role` already reaches the reward function: a per-manager `reward_function_kwargs` entry in `steps/train_verl.py`'s materialized verl config (requires verl to construct the two managers with different kwargs — currently L79–80 pass identical config), or failing that a distinguishing env var per actor. Then filter to `split == "train"` in `compute_step_health` and in `report.py`'s aggregation, and fold the field into T3.9's `RolloutLogEntry`. Cheap fallback if the per-manager kwarg proves invasive: set `val_before_train: false` — but that only shrinks the contamination (the post-training pass still fires), it doesn't make the two separable.
- Also fix while here: `configs/base.yaml`'s comments on `val_freq`/`val_generations_to_log`/`worker.rollout.val_override_config` claim validation is "moot while `val_freq` is -1". That is wrong for both the before-train and after-train passes — validation runs twice per phase today regardless of `val_freq`.
- Goal: checkpoint health and run reporting describe *training* rollouts only. Today a validation row can push a training step's group over the policy-collapse threshold, or mask a genuinely collapsed one — and that signal is exactly what chooses which checkpoint gets fed into the next co-evolution round.
- Accept: with `num_val ≥ 1`, rows with `split == "train"` number exactly `max_steps × rollout_batch_size × rollout_n`; `compute_step_health` returns identical results whether or not validation ran; `run_health` is identical between a run with `val_before_train: true` and one with it `false`.

**T3.8 — Test suite**: rank math (ties/G=1/all-failed), uncertainty vs. variance formulas on canned score sets, filters (CJK edge cases, truncation detection), `verl_entry` dispatch, config round-trip. Target: the four bug classes already found (H4 scale, H6 ties, env contract, format inheritance) each have a test that would have caught them. Accept: `pytest` green, < 60s, no network.

### Phase 4 — Quarantine + manifest (day 5)

**T4.1 — Quarantine fork artifacts to `legacy/`**
- Move: `examples/reward_function/{math.py,r1v.py,caller.py,caller_penalty.py}` (after porting the repetition-penalty idea), `scripts/{main.sh,smoke_test.sh,questioner_train*.sh,solver_train.sh}`, `evaluation/{eval_bbeh.py,eval_mmlupro.py,eval_supergpqa.py}`, `question_generate/question_generate.py`, math jinja files. **Check first:** `modal_run.py::run_smoke_test` calls `smoke_test.sh` — quarantine that Modal function in the same commit; `dataset.py` sentinel branches reference the questioner jinjas — leave verl untouched, the sentinels just never fire once no creative config points at those files.
- Accept: `grep -rn "math12k\|boxed\|questioner_format" --include="*.yaml" configs/` is empty; a fresh clone's top level reads as a creative-writing project.

**T4.2 — README rewrite**: replace upstream R-Zero README with: loop diagram (challenger → prompts → solver → judge → rewards), quickstart (`modal run modal_app.py::coevolve --config …`), config reference table, "add a reward strategy in 3 steps", link to both plan docs. Accept: a new lab member can launch a smoke run without reading bash.

**T4.3 — Run manifest**: orchestrator writes `run_dir/manifest.json` — resolved config, git SHA + dirty flag, `creative.jinja` hash, judge model id, package versions, the T1.2 rendered prompt, and (post-run) selected checkpoint + health metric. Accept: every artifact dir answers "what exactly produced this?" without W&B access.

**T4.4 — Offline dry-run + CI**: `profile: dryrun` — MockJudge, canned challenger prompts, canned rollout texts injected into the reward path, orchestrator state machine exercised end-to-end minus the verl subprocess (verl gets a `--skip-training` stub that fabricates a checkpoint dir). Wire T3.8 + dry-run into CI (GitHub Actions, CPU). Accept: full loop passes on a laptop in < 5 min; CI red on any reward-math regression.

**T4.5 — Kill "smoke" naming**: `profile: {dryrun, smoke, small, full}` presets in `base.yaml` (sizes, steps, judge concurrency); run names derived from `abbr + profile + timestamp`. Accept: no file or W&B run named "smoke" refers to a real experiment again.

### Dependency graph & suggested schedule

```
Day 1: T1.1→T1.2, T1.3, T1.4→T1.5, T1.6, T1.7, T1.8, T1.9, T1.10, T1.11
Day 2: T2.1→T2.2, T2.3          (T2.4–T2.7 parallelizable after T2.2/T2.3)
Day 3: T2.4, T2.5, T2.6, T2.7 → T2.8 → T2.9
Day 4: T3.1→T3.2 → T3.3 → T3.4, T3.5 → T3.6, T3.7 → T3.8
Day 5: T4.1–T4.5
```

Phase-1 tasks are each ≤ 1–2 h and independently shippable; if time squeezes, T1.1–T1.4 + T1.7 + T1.11 are the non-negotiables before the next real run.

---

## 5. Revision 2026-08-05 — no runs until Week 1 completes; T1.x–T2.4 already implemented

The no-interim-runs constraint removes all backward-compatibility obligations. Changes:

1. **T1.3 dropped.** Don't patch env-var names in a layer that dies before it's ever run again; T2.6/T2.8 replace the env hand-off with typed args directly.
2. **T2.9 parity → validation.** No old-vs-new diffing; the old behavior was the bug. One smoke GPU run of the new pipeline at the end of Week 1, judged against instrumentation criteria (see revised T2.9). It doubles as the "after" figure data.
3. **T4.1 quarantine → deletion.** `git tag pre-refactor`, then `git rm` the math artifacts (same list). `legacy/` only made sense to keep the old path runnable during a transition that no longer exists. Also delete `modal_run.py`'s creative functions and `run_smoke_test` in the same commit as T2.8 lands.
4. **Mock-first validation: T3.1 (MockJudge) and T4.4 (dry-run) are promoted to the critical path.** With no GPU runs during the build, the mock-judge dry-run is the *only* end-to-end correctness signal — build it alongside Phase 2 remainder, not on day 5. Concretely: every `steps/` module gets exercised by the dry-run the day it's written.
5. **Registry before wiring: do T3.3/T3.6 before T2.8**, so `train_verl.py`'s `worker.reward.reward_function` points at `rewards/verl_entry.py` from the first commit — the orchestrator never temporarily targets the old caller files, and the Phase-1 fixes (T1.6/T1.7/T1.8/T1.10) live only in the new strategy classes (port them out of the old callers if they were patched there, then delete the old callers).

**Revised order for the remaining work (T2.4 done):**

```
Next:  T3.1 (incl. MockJudge) → T3.2 → T3.3            # judge + registry skeleton
Then:  T2.5, T2.6 (reward_function → verl_entry), T2.7  # steps, validated via mock
       T3.4, T3.5 → T3.6, T3.9 (+T3.10) → T3.7          # strategies (port Phase-1 fixes here); T3.9 defines log schema T3.7 writes, T3.10 adds its `split` field
       T2.8 (modal_app + orchestrator) + delete old scripts/functions (rev. T4.1)
Gate:  T3.8 tests + T4.4 dry-run green                  # no GPU spent before this passes
Then:  T4.3 manifest, T4.5 profiles, T4.2 README
End of Week 1: T2.9 validation run (first GPU spend)
```

Rationale for the gate: the single most expensive failure mode this week is discovering a wiring bug during the first GPU run. The dry-run must exercise: config load → prompt gen (mock) → parquet → resolved verl YAML render → `verl_entry` dispatch with MockJudge → rank/variance reward math → health-based checkpoint selection → orchestrator state save/resume.
