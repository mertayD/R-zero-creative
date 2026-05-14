---
name: eval_postprocess
description: Inspect and compare WritingBench eval outputs from R-Zero runs — pair model responses with prompts and judge scores, find worst-performing prompts, diff scores across iterations. Use this skill whenever the user asks to "see the model's responses", "inspect what the base model wrote", "look at outputs from the eval", "find which prompts scored low", "compare iter1 vs iter2", "look at the judge's reasoning", "what failed on the smoke run", or anything involving examining the artifacts produced by scripts/eval_writing_bench.sh (responses.jsonl, scores/*.jsonl, scores.xlsx). Also triggers on phrases like "pull examples for the paper", "show me a few qualitative outputs", "why did the score drop", "WritingBench inspection", or any request to dig into the contents of eval_samples / writing_bench output directories.
---

# eval_postprocess

Helpers for inspecting WritingBench eval artifacts after `scripts/eval_writing_bench.sh` finishes. The eval pipeline writes three files per run; this skill joins them by `index` so you can see prompt → response → judge scores side by side.

## Run directory layout

After an eval run, artifacts live at:

```
<storage_path>/writing_bench/<model_slug>/<subset>/
├── responses.jsonl                  {"index": i, "response": "<full text>"}
├── scores/<model_slug>.jsonl        {"index": i, "scores": {criterion: [{score, reason}]}}
└── scores.xlsx                      aggregated rollups (Overall, Domain1_*, ...)
```

The subset name comes from the directory: `smoke50`, `mid100`, or `full`. The corresponding prompt + criteria file is:

- `smoke50` → `evaluation/writing_bench/subsets/smoke50.jsonl`
- `mid100`  → `evaluation/writing_bench/subsets/mid100.jsonl`
- `full`    → `evaluation/writing_bench/benchmark_query/benchmark_all.jsonl`

## When to use which command

The bundled `inspect.py` exposes four subcommands. Pick by what the user is trying to learn.

### "Show me what the model wrote for prompt N" → `one`

Print the prompt, the full response, and the per-criterion scores + reasons for a single index. Use when debugging why a specific prompt scored low or pulling a qualitative example for a writeup.

```bash
python evaluation/eval_postprocess/inspect.py one \
    --run <run_dir> --index 42
```

### "Give me a table of every prompt and its average score" → `list`

One row per prompt: index, domain, language, average score, per-criterion scores. Sortable by score. Use as the entry point for any "which prompts are doing well or badly" question.

```bash
python evaluation/eval_postprocess/inspect.py list \
    --run <run_dir> [--sort score|index] [--domain "Education"] [--lang en|zh]
```

### "What are the worst N prompts" → `worst`

Shortcut for `list --sort score | head`. Prints the lowest-scoring prompts with their per-criterion breakdown. Most useful for "where is the model failing".

```bash
python evaluation/eval_postprocess/inspect.py worst \
    --run <run_dir> --n 10
```

### "How did iter1 change vs. base model" → `compare`

Diff two run directories on the same subset. Prints per-prompt delta and the overall mean shift. This is the workhorse for tracking R-Zero progress across iterations.

```bash
python evaluation/eval_postprocess/inspect.py compare \
    --run-a <base_run_dir> --run-b <iter1_run_dir>
```

## Conventions the script relies on

- The run directory name **is** the subset name (`smoke50` / `mid100` / `full`). If a user has renamed a run dir, pass `--subset` explicitly.
- The score file inside `<run_dir>/scores/` is named `<parent_dir_name>.jsonl` (i.e., the model slug). The script picks the single `.jsonl` in that folder, so renaming is fine.
- Prompts and responses can be very long (16k tokens). The default truncates to 2000 chars per field for readability; pass `--full` to print everything.

## Programmatic use

The script also exposes a Python API for use in notebooks or analysis scripts:

```python
from evaluation.eval_postprocess.inspect import load_run, join_run

records = load_run("/storage/writing_bench/Qwen_Qwen3-4B-Base/smoke50")
# records is a dict[index] -> {"prompt", "domain1", "domain2", "lang",
#                              "checklist", "response", "scores", "avg"}

import pandas as pd
df = pd.DataFrame(join_run(records))
df.sort_values("avg").head(5)
```

Use `load_run` when you need the full structured data (judge reasons, all criteria); use `join_run` when you want a flat list-of-dicts suitable for a DataFrame.

## What this skill does NOT do

- It does not re-judge responses. Scores are read as-is from the run artifacts.
- It does not modify `scores.xlsx`. That's the official WritingBench aggregation; for new aggregations write your own.
- It does not aggregate across many runs (e.g., "leaderboard across all my checkpoints"). For that, parse multiple `scores.xlsx` files with pandas — out of scope here.
