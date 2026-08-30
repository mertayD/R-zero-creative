# Challenger eval harness

A **standalone** harness for asking "how good is this challenger checkpoint
right now?" — independent of the orchestrator and `configs/base.yaml`. It
doesn't train anything or feed back into co-evolution; it just takes a
checkpoint path and produces a report (files on disk, plus a best-effort W&B
log — see "W&B logging" below).

Two phases, run separately:

1. **`build_dataset.py`** — build the frozen eval set once, upload it to the
   HF Hub.
2. **`run_eval.py`** — point at any challenger checkpoint, generate against
   that frozen set, score a full metric battery, write a report.

## Phase 1: build the eval set

Every `(domain, subdomain)` pair from `WRITING_DOMAINS` (6 domains, 76 pairs)
× `REPLICATES_PER_PAIR` (10) replicates, each replicate an independently
sampled `guidance_text` subset — 760 rows total. Built with the exact same
`build_one_shot_prompt` the challenger's own training pipeline uses
(`creative_rzero/steps/generate_prompts.py`), so the eval prompts are
identical in shape to what the challenger sees during training.

The set is frozen by construction: `random.seed(f"{domain}|{subdomain}|
{replicate_idx}")` before every `build_one_shot_prompt` call makes each row's
guidance draw reproducible across re-runs — the same 760 rows every time you
run `build_dataset.py`, not a fresh random draw. That matters because
`run_eval.py` is meant to be run repeatedly against different checkpoints;
if the input set changed between runs, a metric swing could just be eval-set
noise instead of a real change in the challenger.

```bash
# Inspect the set locally first, no Hub credentials needed
python -m creative_rzero.eval.challenger.build_dataset --dry-run

# Push for real (needs HF_TOKEN + HUGGINGFACENAME in env)
python -m creative_rzero.eval.challenger.build_dataset --repo_name challenger-eval-v1
```

Only rebuild/push a new version if the eval set itself needs to change
(e.g. `REPLICATES_PER_PAIR` changes, or `WRITING_DOMAINS`/the guidance pool
changes upstream) — otherwise every checkpoint eval should point at the same
Hub repo so results stay comparable over time.

## Phase 2: run against a checkpoint

```bash
python -m creative_rzero.eval.challenger.run_eval \
    --model /path/to/challenger/checkpoint/huggingface \
    --dataset-repo <HUGGINGFACENAME>/challenger-eval-v1 \
    --judge-type claude \
    --out-path /tmp/challenger_eval/step_120.jsonl
```

Writes two files: a per-row JSONL (`--out-path`) and a stratified summary
(`<out-path>.summary.json`, from `aggregate.py`) with overall / per-domain /
per-subdomain breakdowns.

### Run on Modal

`modal_app.py::run_challenger_eval` runs the same harness on a GPU box on
Modal and `modal_app.py::challenger_eval` is its local entrypoint — call it
with `.remote()` (not `.spawn()`), so the CLI blocks and prints the summary
as soon as the run finishes instead of returning a function-call id you'd
have to poll separately:

```bash
modal run modal_app.py::challenger_eval \
    --checkpoint /storage/models/<run>/global_step_120/actor/huggingface \
    --step 120
```

`checkpoint` is anything vLLM can load — a merged checkpoint path on the
Modal volume or a bare HF model id. `--dataset-repo` defaults to
`HUGGINGFACENAME/challenger-eval-v1` (`CHALLENGER_EVAL_DATASET_REPO` in
`modal_app.py`, override via env). Results also land on the volume under
`$STORAGE_PATH/eval/challenger/results/`. Runs on `CHALLENGER_EVAL_GPU`
(default `A100-40GB:1` — one GPU is enough since this is vLLM inference, no
training) rather than the training `CREATIVE_GPU` spec.

## W&B logging

`run_eval.py::run()` logs to W&B whenever `wandb` is importable and
`WANDB_MODE` isn't `"disabled"` (mirrors `generate_prompts.py::
run_generation`'s guard) — it's best-effort, not required: the function
still returns the summary dict either way. If `WANDB_RUN_ID` is set (e.g.
this eval is invoked from inside an active training run) it attaches to
that run instead of creating its own. Logged per call:

- `challenger_eval/overall/*` and `challenger_eval/by_domain/<D>/*` —
  flattened scalar metrics (format pass rate, judge-score means, duplicate
  rate, length stats, `format_failure_reason` breakdown) so they're
  chartable directly in the W&B UI.
- `challenger_eval/by_subdomain` — the same stats per subdomain, as a Table
  rather than ~600 scalar metrics.
- `challenger_eval/rows` — one Table row per evaluated prompt, for drilling
  into a specific `eval_id`.

Pass `--step` (CLI) / `step=` (`run()`) when you want repeated eval runs
against different checkpoints to line up on a shared x-axis in W&B — it's
plumbed straight to `wandb.log(metrics, step=step)`, not derived
automatically from the checkpoint path.

## What gets scored, and why

- **Formatting** — reuses `generate_prompts.py::validate_one_shot_response`
  and the `FormatFailureReason` taxonomy unchanged. No new validation logic;
  this is the same check the training pipeline's retry loop already applies.
- **Domain adherence / guidance adherence / criteria quality** — a single
  Claude judge call per format-valid row (`judge_prompts.py`), bundled into
  one call rather than three to keep judge cost down. Formatting can pass
  while the query ignores its assigned subdomain or the guidance principles
  it was supposed to apply — this catches that. `judge_prompts.py` is a new
  prompt, distinct from `evaluation/writing_bench/batch_eval_prompt.py`:
  that one scores a *solver response* against criteria; this one judges the
  *challenger's own generated prompt* before any solver sees it.
- **Diversity / near-duplicates** — `diversity.py`, run per
  `(domain, subdomain)` group (10 replicates), not across the full 760-row
  set. Comparing across unrelated subdomains is low-signal — they're
  *supposed* to look different. Near-identical queries *within* one pair's
  10 replicates, despite each replicate having a different sampled
  `guidance_text`, is the real mode-collapse signal: the guidance was
  supposed to push each one in a different direction. TF-IDF + cosine
  similarity (`scikit-learn`, already a repo dependency), English stopwords
  removed, threshold 0.32 — both calibrated against blind human-protocol
  judgment of 1,243 pairs (see `diversity.py`'s docstring).
- **Length** — `query_len`/`criteria_len` per row, mean/stddev in the
  aggregate. Cheap, no judge call, often the first visible symptom of
  collapse or reward hacking (queries/criteria drifting shorter, or toward
  boilerplate).

## Judge backends

`--judge-type claude` is the real judge (`evaluator/llm.py::ClaudeAgent`,
needs `PERPLEXITY_API_KEY`). `--judge-type mock`
(`evaluator/mock.py::MockJudgeAgent`) is a **smoke-test-only** path — no API
key, deterministic fake scores, useful for e.g. `run_eval.py --limit 10`
to prove the generate → judge → parse → aggregate plumbing doesn't crash
before spending real judge calls on 760 rows.

Caveat: `MockJudgeAgent.run()` looks for `"Name: X"` lines in the prompt (the
shape WritingBench's `BatchEvalAgent` bakes in for scoring several criteria
in one call) and only falls back to a single `{"score", "reason"}` object
when none are found. This harness's judge prompt has no such lines, so mock
mode always hits that fallback — never the harness's real 3-field schema.
`challenger_judge_agent._parse` detects this and duplicates the one mock score into all
three fields, tagging the row `judge_backend: "mock"` so it's never confused
with a real per-dimension score. Don't draw conclusions about domain/
guidance/criteria-quality from a mock run — only use it to check the harness
itself runs end to end.

`--judge-type sft-critic` is **not offered**. `CriticServerAgent` /
`PerCriterionEvalAgent` is a model fine-tuned to score a *solver response*
against *criteria* (the WritingBench task); judging whether a *generated
prompt* fits its assigned domain/guidance is a different task it was never
trained on, so wiring it in here would silently misuse it.
