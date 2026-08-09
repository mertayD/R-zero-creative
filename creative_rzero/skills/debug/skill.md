---
name: debug-run
description: Diagnose a co-evolution training run on Modal — empty or zero-reward rollouts, PromptShortfallError, merge/checkpoint failures, judge errors, or "the run died and I don't know why". Use when investigating any r-zero-creative run by its abbr/run_ts, reading reward_logs JSONL, or deciding whether a failure is the challenger, the solver, the judge, or infrastructure.
---

# Debugging a co-evolution run

A run is one `modal run modal_app.py::main` invocation. It alternates two phases
per iteration, and **almost every failure is diagnosable from artifacts on the
Modal volume without re-running anything.** Read the artifacts first; only fall
back to container logs for stack traces.

```
Phase A (challenger)         Phase B (solver)
build challenger parquet     generate prompts from challenger ckpt  <- subprocess, aux GPU
start vLLM solver oracle     build solver parquet
verl GRPO training           verl GRPO training
merge checkpoint             merge checkpoint
upload rollout log           upload rollout log
```

Reward flow: challenger writes a prompt -> vLLM solver oracle answers it ->
Claude judge scores the answer against the challenger's criteria -> R-Zero
uncertainty reward. **Three services must all work for a nonzero reward.** The
`failure_reason` column tells you which one broke — do not guess.

## Fastest path: the run's W&B page

`creative_rzero/steps/report.py` consolidates every artifact below into **one
W&B run per co-evolution run** — id `{abbr}_{run_ts}`, name
`coevolve_{abbr}_{run_ts}`, project `r-zero-creative`. Both GPU phase
containers and the CPU orchestrator resume that same run, so there is one page
to open rather than four.

| W&B key | What it shows |
|---|---|
| `run_health` | one row per (iteration, role): n_rollouts, ok/infra/policy failure rates, format_valid_rate, mean reward, top failure reason |
| `{role}/rollouts` | every rollout row, including full `raw_output` and `format_reason` |
| `{role}/failure_breakdown` | `failure_reason` and `format_reason` counts with shares |
| `{role}/step_health` | per-global_step policy-collapse rate, hard/soft flags |
| `prompt_gen/failures` | Phase B failed attempts with full raw responses |
| `prompt_gen/breakdown` | Phase B failure-reason counts |

Read `run_health` first — the `infra_failure_rate` vs `policy_failure_rate`
split answers "my problem or the judge's?" before you open anything else.
Per-phase scalars are also written to run summary
(`{role}/iter{N}/{metric}`), so the W&B runs list sorts and compares health
*across* runs.

Reports are logged **before** merging, so a run that dies in merge still has
its tables. Everything below is the same data at the source — use it when W&B
is unavailable, or when you need the raw bytes.

## Step 0 — identify the run

Everything is keyed on `{abbr}_{run_ts}_iter{N}`. Get `run_ts` from the launch
output (`[FRESH] Starting co-evolution run_ts=...`) or by listing artifacts:

```bash
modal volume ls r-zero-storage reward_logs | tail -5   # newest runs
modal app list                                          # app state, running vs stopped
```

The volume is `r-zero-storage` (override: `VOLUME_NAME` in `.env`), mounted at
`/storage` in containers. `creative_rzero/paths.py` is the authority for layout.

## The volume map

| Path | What it answers |
|---|---|
| `reward_logs/{abbr}_{ts}_iter{N}_{role}_v1.jsonl` | per-rollout truth: what the model wrote, what failed |
| `generated_questions_one_shot/..._solver_prompts_{profile}.json` | Phase B output + `generation_log` counts |
| `generated_questions_one_shot/..._solver_prompts_{profile}.failures.jsonl` | Phase B failed attempts, full raw responses |
| `creative_smoke/..._{train,val}.parquet` | exact training rows (challenger: no `solver_` infix) |
| `models/..._{role}_v1/global_step_N/` | what verl actually saved (**1-indexed**) |
| `models/..._{role}_last_ckpt.txt` | merged checkpoint path the phase reported |
| `coevolve_state/{abbr}.json` | resume state — keyed on abbr ALONE, not run_ts |
| `runs/{abbr}_{ts}/manifest.json` | resolved config + git SHA for the run |

Download and inspect:

```bash
modal volume get --force r-zero-storage reward_logs/<file>.jsonl .
modal volume ls r-zero-storage models/<save_name>          # what steps exist
modal app logs <app-id-or-name>                            # stack traces only
```

## Step 1 — always start here: the failure_reason histogram

One command answers "which of the three services broke":

```bash
jq -r '[.format_valid, .format_reason, .failure_reason] | @tsv' *_challenger_v1.jsonl \
  | sort | uniq -c | sort -rn
```

Read the result against this table. `format` in the reward dict is **not** a
format quality score — it is `1.0` on every path where parsing succeeded but
scoring failed, so `format: 1.0, overall: 0.0` means "challenger fine, something
downstream broke", never "the challenger did well".

| failure_reason | Who broke | First move |
|---|---|---|
| `ok` | nothing | reward should be nonzero; if not, inspect `raw_score`/ranking |
| `truncated` | challenger, out of budget | raise role `max_response_length` |
| `challenger_format_invalid` | challenger output | read `format_reason` + `raw_output` |
| `solver_api_error` | vLLM oracle | check the oracle started and stayed healthy in logs |
| `empty_answer` | solver | oracle responded but produced nothing |
| `invalid_criteria` | challenger data | criteria missing required fields |
| `judge_parse_fail` | judge | responded but output never parsed |
| `judge_rate_limit` / `judge_api_error` | judge/infra | probe the gateway directly (Step 4) |

`judge_api_error` and `judge_rate_limit` are the only two in
`INFRA_FAILURE_REASONS` — checkpoint health scoring treats them as "not the
policy's fault", everything else counts as policy collapse.

When `format_valid == 0`, `format_reason` (enum `FormatFailureReason` in
`creative_rzero/failure_reasons.py`) says exactly which check failed:
`missing_json_fence`, `invalid_json`, `top_level_not_dict`, `query_not_string`,
`non_english_query`, `missing_query_or_criteria`, `criteria_not_list`,
`criterion_not_dict`, `criterion_missing_fields`,
`criterion_missing_score_level`, `writing_prompt_fields`,
`empty_query_or_criteria`.

## Step 2 — prove it with one row

Aggregate counts tell you where; a single row tells you why. The challenger log
carries the full untruncated model output, so you never have to re-run to see
what happened.

```bash
# widest useful view: is the challenger producing? is the solver producing?
echo "step idx reason solver_len think_len query_len crit_n format overall"
jq -r '[.step, .rollout_idx, .failure_reason, (.solver_response|length),
        (.solver_thinking|length), (.generated_query|length),
        .generated_criteria_n, .format, .overall] | @tsv' *_challenger_v1.jsonl

# the raw model output, rendered readably (newlines expanded)
jq -r 'select(.format_valid==0) | .raw_output' *_challenger_v1.jsonl | head -c 3000

# every field of one row as labeled blocks
sed -n '1p' *_challenger_v1.jsonl | jq -r 'to_entries[] | "-- \(.key) --\n\(.value)\n"'
```

Interpreting the widths: `solver_response` and `generated_query` are 300-char
previews, so `300` means "at least 300 chars" (healthy) and `0` means genuinely
empty. `raw_output` and both thinking traces are full length.

Challenger row schema: `step`, `rollout_idx`, `domain`, `domain_name`,
`subdomain`, `input_prompt`, `raw_output`, `format_valid`, `format_reason`,
`challenger_thinking`, `generated_query`, `generated_criteria_n`,
`solver_response`, `solver_thinking`, `overall`, `format`, `accuracy`,
`failure_reason`.

Solver row schema: `step`, `prompt_id`, `domain`, `domain_name`, `subdomain`,
`num_criteria`, `input_query`, `response_preview`, `raw_score`,
`failure_reason`, `truncated`, `is_low_quality`, `rank_reward`, `accuracy`.

## Step 3 — Phase B (PromptShortfallError)

Phase B regenerates prompts from the merged challenger checkpoint. If it can't
produce `solver.num_train + num_val` valid prompts, the run dies there. The
error message already embeds the reason counts; the artifacts have the rest:

```bash
jq '{n: (.prompts|length), generation_log}' *_solver_prompts_*.json
jq -r '.failure_reason' *_solver_prompts_*.failures.jsonl | sort | uniq -c | sort -rn
jq -r 'select(.failure_reason=="invalid_json") | .raw_response' *.failures.jsonl | head -c 2000
```

Note the regime difference: Phase B generation runs with **thinking enabled and
a 32k budget**, while training rollouts run with whatever the config sets (often
thinking off, small budget). Format rates legitimately differ between the two —
if they diverge sharply, that asymmetry is the first suspect.

## Step 4 — judge failures

The judge posts to the Perplexity Router gateway
(`https://api.perplexity.ai/router/v1/messages`) with `PERPLEXITY_API_KEY`, and
retries `judge.http_max_retry_attempts` times before raising. Container logs get
rate-limited and the per-attempt `HTTP <status>` lines are often dropped — so
probe the gateway directly instead of hunting for them:

```bash
.venv/bin/python3.11 -c "
from dotenv import load_dotenv; import os, requests
load_dotenv()
k = os.environ['PERPLEXITY_API_KEY']
r = requests.post('https://api.perplexity.ai/router/v1/messages',
  headers={'Authorization': f'Bearer {k}', 'anthropic-version':'2023-06-01',
           'content-type':'application/json'},
  json={'model':'anthropic/claude-sonnet-5','max_tokens':16,
        'messages':[{'role':'user','content':'say ok'}]}, timeout=60)
print('HTTP', r.status_code); print(r.text[:400])
"
```

401/403 is credentials or account access, not your code. To keep validating the
rest of the pipeline without judge spend, set `judge.type: mock`.

Never print the key itself — check presence with `len()` and a short prefix.

## Step 5 — checkpoint / merge failures

`MergeFailedError ... FileNotFoundError: .../global_step_N/actor` means the step
selector chose a step verl never wrote. Compare intent against reality:

```bash
modal volume ls r-zero-storage models/<save_name>       # global_step_1, global_step_2, latest_global_step.txt
modal volume get --force r-zero-storage models/<save_name>/latest_global_step.txt .
```

**verl writes 1-indexed checkpoint dirs** (`global_step_1` … `global_step_{max_steps}`).
`merge_ckpt.py` selects via `compute_step_health` / `select_checkpoint_step`,
which map the reward log's 1-indexed call counter to a 0-indexed step and search
`range(max_steps)`. When the two conventions disagree, merge asks for a
directory that cannot exist. Check this before assuming training failed to save.

Step selection also depends on health: a step needs collapse rate
< `HEALTHY_RATE` (0.10) and no flags to be preferred, else the least-bad step
wins. Because `_is_policy_failure` counts everything except the two infra
reasons, a batch full of `truncated` or format failures can push selection to an
earlier step.

## Preflight and cheap validation

```bash
modal run modal_app.py::preflight_check --config configs/exp/validation_tiny.yaml
.venv/bin/pytest tests/ -q
```

`preflight_check` is CPU-only: it validates config, materializes the verl config
for both roles, imports the reward entrypoint the way verl does, and constructs
the judge client. It catches wiring breakage before GPU spend — but note it only
*constructs* the judge client, so it will not catch a credential that is present
but unauthorized.

## Gotchas that waste hours

- **Stale resume state silently skips Phase A.** `coevolve_state/{abbr}.json` is
  keyed on `abbr` alone. A new run with the same abbr resumes mid-run — you get
  no challenger rollouts and the same downstream failure. Stop the app first,
  then `modal volume rm r-zero-storage coevolve_state/{abbr}.json`. The file only
  exists once Phase A has completed; "file not found" is harmless.
- **Modal ships your working tree, not a commit.** `add_local_dir(".", copy=False)`
  means uncommitted edits go to the container and a *running* app keeps the code
  it launched with. A fix does not apply to an in-flight run.
- **Container logs are rate-limited.** Expect gaps ("output rate limiting
  expired"). Never conclude "no errors" from a silent grep — use the JSONL.
- **The reward `format` field is not a quality signal.** It is `1.0` on every
  post-parse failure path.
- **Zero reward everywhere means zero GRPO advantage.** The model cannot learn
  from a uniformly-zero batch, so a 100% failure rate is a wiring bug to fix, not
  a result to interpret.
- **Artifacts are per-`run_ts`, state is per-`abbr`.** Old runs' logs and
  checkpoints never collide and never need cleanup.

## Worked example

Symptom: all challenger rollouts empty, then `PromptShortfallError`.

1. Histogram: 100% `missing_output_tags`, zero `truncated` -> not a budget problem.
2. One raw row: the model had produced complete, schema-correct JSON wrapped in
   ` ```json ` fences — the parser wanted `<output>` tags. Fixed by moving the
   contract to fences (prompt copy + `FormatValidator` + truncation heuristic).
3. Re-run: format-valid went 0% -> 90%, and the histogram moved to
   `judge_api_error`. Direct gateway probe returned **HTTP 403 "Router API is
   currently in limited preview"** — a credential/access problem, not the solver.
   The rollout rows proved the solver was healthy (300-char responses, 1–2.7k-char
   thinking traces), which is exactly what `empty_answer` / `solver_api_error`
   would have contradicted.
4. The run then died in merge asking for `global_step_0` while the volume held
   `global_step_1` and `global_step_2` — the 1-indexed/0-indexed mismatch above.

Three independent bugs, all diagnosed from volume artifacts, no re-runs needed.
