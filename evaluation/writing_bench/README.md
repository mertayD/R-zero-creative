# WritingBench evaluation

Drop-in WritingBench eval for the no-RL Qwen3-4B-Base baseline (and any later
R-Zero checkpoint). Scores produced here are directly comparable to the public
[WritingBench leaderboard](https://huggingface.co/spaces/WritingBench/WritingBench)
because we use:

  * the **official prompts and criteria** from `benchmark_query/benchmark_all.jsonl`
  * the **official scoring template** from `prompt.py` (vendored verbatim from
    [X-PLUG/WritingBench](https://github.com/X-PLUG/WritingBench), Apache-2.0)
  * the **official generation params** — `top_p=0.8, top_k=20, temperature=0.7,
    max_tokens=16000` — per the WritingBench README
  * the **official judge** — Claude-Sonnet-4-5 via Anthropic's Messages API,
    matching the leaderboard switch on 2025-11-27

The development-time judge for R-Zero training (a base-model judge) is **not**
implemented here — that lives outside the eval path on purpose.

## Layout

```
evaluation/writing_bench/
├── README.md                       (this file)
├── fetch_data.sh                   one-shot clone of upstream benchmark_query/
├── make_subsets.py                 reproducible smoke50 / mid100 sampler
├── subsets/
│   ├── smoke50.jsonl               50 prompts, stratified by domain1 (committed)
│   └── mid100.jsonl                100 prompts, stratified by domain1 (committed)
├── benchmark_query/                fetched on demand, .gitignored
│   ├── benchmark_all.jsonl
│   └── requirement/{style,format,length}/...
├── generate_responses_vllm.py      vLLM batch driver (replaces upstream stub)
├── prompt.py                       vendored verbatim
├── evaluate_benchmark.py           vendored verbatim
├── calculate_scores.py             vendored (one missing typing import added)
└── evaluator/
    ├── __init__.py
    ├── critic.py                   vendored verbatim — local Qwen-7B critic path
    └── llm.py                      patched: Anthropic Messages API + ANTHROPIC_API_KEY
```

## Three subset choices

| Subset    | Size | Use                                                    |
| --------- | ---- | ------------------------------------------------------ |
| `smoke50` | 50   | End-to-end smoke; sanity-check the pipeline.           |
| `mid100`  | 100  | Between-iteration evals during R-Zero training.        |
| `full`    | 1000 | Headline number for papers / leaderboard comparison.   |

Both `smoke50` and `mid100` are stratified by `domain1` (six categories) and
seeded so they reproduce exactly across machines. The full benchmark is fetched
by `fetch_data.sh`.

## One-time setup

```bash
# Pull the WritingBench data (~40 MB). Optional — the runner auto-fetches on
# first invocation if benchmark_query/ is missing.
bash evaluation/writing_bench/fetch_data.sh

# Required env: Anthropic key for the Claude-Sonnet-4-5 judge.
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run the baseline

The runner takes three positional args: model, subset, and the storage path
where outputs are written (`<storage_path>/writing_bench/<model_slug>/<subset>/`).
The storage path should point OUTSIDE this repo (e.g. `~/wb-runs` or a scratch
disk) — eval artifacts can grow large quickly across iterations.

```bash
# Smoke (50 prompts, ~minutes of generation + ~50*5 = 250 judge calls).
bash scripts/eval_writing_bench.sh Qwen/Qwen3-4B-Base smoke50 ~/wb-runs

# Mid-training subset (100 prompts).
bash scripts/eval_writing_bench.sh Qwen/Qwen3-4B-Base mid100 /data/wb

# Full leaderboard-comparable eval.
bash scripts/eval_writing_bench.sh Qwen/Qwen3-4B-Base full ~/wb-runs
```

If you already export `STORAGE_PATH` for the rest of R-Zero, you can omit the
third arg and the runner will fall back to it.

Outputs (rooted at the storage path you passed):

```
<storage_path>/writing_bench/<model_slug>/<subset>/
  responses.jsonl                 # {index, response} per query
  scores/<model_slug>.jsonl       # {index, scores: {criterion: [{score, reason}]}}
  scores.xlsx                     # Overall + per-domain1/domain2 + style/format/length
```

## Plugging in a new checkpoint

The runner takes any HF id or local path:

```bash
bash scripts/eval_writing_bench.sh "$STORAGE_PATH/models/qwen3-4b-iter1" mid100 ~/wb-runs
bash scripts/eval_writing_bench.sh "$STORAGE_PATH/models/qwen3-4b-iter2" mid100 ~/wb-runs
# ...
```

`scores.xlsx` files from multiple runs can be diffed directly because the
column schema is identical.

## Notes on judge choice

- We default to **Claude-Sonnet-4-5** (`claude-sonnet-4-5`) because the public
  WritingBench leaderboard switched to it on 2025-11-27. To override, set
  `WB_JUDGE_MODEL` (e.g. `claude-3-7-sonnet-20250219`).
- The local critic model path (`AQuarterMile/WritingBench-Critic-Model-Qwen-7B`)
  is wired but disabled by default. To use it, edit `evaluator/critic.py` to
  set the local model path and call `evaluate_benchmark.py --evaluator critic`.

## Re-generating the subsets

```bash
python evaluation/writing_bench/make_subsets.py \
    --benchmark_all evaluation/writing_bench/benchmark_query/benchmark_all.jsonl \
    --out_dir evaluation/writing_bench/subsets
```

This will overwrite the committed `smoke50.jsonl` and `mid100.jsonl`. Don't run
it casually — if the subset changes, prior runs are no longer comparable.
