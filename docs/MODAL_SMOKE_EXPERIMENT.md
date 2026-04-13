# Modal smoke experiment — guide and findings

This document describes the **R-Zero end-to-end smoke run** on **Modal**: how to start it, the **workflow** inside the container, **storage**, and **issues we hit** (and how they were fixed). Use it for handoff, reruns, and debugging.

---

## What this is (and is not)

- **Goal:** Run **one** full R-Zero iteration (questioner → merge → generate / evaluate / upload → solver → merge → optional GSM8K check) on **small data** to validate the pipeline before scaling.
- **Implementation:** [`scripts/smoke_test.sh`](../scripts/smoke_test.sh) mirrors the **stages** of the paper/README loop. [`scripts/main.sh`](../scripts/main.sh) runs **five** iterations; smoke runs **one**.
- **Modal glue:** [`modal_run.py`](../modal_run.py) defines the image, volume, secrets, and a single GPU function that invokes `smoke_test.sh`. The upstream R-Zero repo does not ship Modal code.

---

## Prerequisites

1. **Modal CLI** logged in; workspace/profile for this project (e.g. `MODAL_PROFILE=columbia-daplab` in [`.env`](../.env)).
2. **`.env`** (gitignored): `APP_NAME`, `VOLUME_NAME`, `MODAL_FULL_GPU`, `REMOTE_REPO_PATH`, `REMOTE_STORAGE_PATH`, **`HUGGINGFACENAME`** (HF username/org for dataset repos — **required** for solver upload + `data.train_files`), timeouts. [`modal_run.py`](../modal_run.py) loads it with `python-dotenv` **locally**.
3. **`tokens.json`** (local, gitignored): `huggingface` and `wandb` keys. Read **only when `modal.is_local()`**; the container gets **`HF_TOKEN`** / **`WANDB_API_KEY`** from the Modal secret.
4. **Modal secret / env:** At runtime the container needs **`HUGGINGFACENAME`**, **`STORAGE_PATH`**, and HF/W&B tokens. `modal_run.py` builds `modal.Secret.from_dict` from local `tokens.json` + env when you run from your machine.
5. **Volume:** `modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)` — artifacts under **`REMOTE_STORAGE_PATH`** (default **`/storage`**).

---

## How to start a run

**Always use `--detach`** for long GPU jobs so losing the local terminal or network does not kill the remote job.

```bash
cd /path/to/R-Zero
modal run --detach modal_run.py
```

**With CLI overrides** (Hydra-style args on the Modal entrypoint):

```bash
modal run --detach modal_run.py \
  --base-model Qwen/Qwen3-4B-Base \
  --abbr qwen3-4b-smoke \
  --smoke-max-train-samples 8 \
  --smoke-max-val-samples 8 \
  --smoke-rollout-batch-size 8 \
  --smoke-val-batch-size 8 \
  --smoke-global-batch-size 8 \
  --smoke-questioner-max-steps 2 \
  --smoke-solver-max-steps 4 \
  --smoke-questions-per-gpu 8
```

**Thin wrapper:**

```bash
bash scripts/modal_run_smoke_detach.sh
```

After `modal run` prints **“View run at”**, open that **Modal dashboard** URL for live logs. Local streaming may continue until the function exits; with **`--detach`**, stopping the local process should **not** stop the remote function.

**Logs from CLI** (app name from `.env` `APP_NAME`, default `r-zero-main`):

```bash
modal app list
modal app logs r-zero-main
```

---

## End-to-end workflow (what runs inside the container)

`modal_run.py` sets `HF_HOME` / `HUGGINGFACE_HUB_CACHE` under **`/storage/hf_cache`**, ensures dirs (`evaluation`, `models`, `generated_question`, `temp_results`), then runs:

```bash
bash scripts/smoke_test.sh <base_model> <abbr>
```

### STEP 1 / 2 — Questioner v1

1. **`RUN_ID`** exported; four **vLLM** reward servers start on **GPUs 4–7** ([`vllm_service_init/start.sh`](../vllm_service_init/start.sh)), ports **5000–5003**.
2. Smoke **polls `http://127.0.0.1:<port>/health`** until all are up (reward code must use **127.0.0.1**, not `0.0.0.0`, for reliable `requests`).
3. **GRPO training** on **GPUs 0–3** (`trainer.n_gpus_per_node=4`), reward [`caller_penalty.py`](../examples/reward_function/caller_penalty.py), `trainer.max_steps=SMOKE_QUESTIONER_MAX_STEPS` (default **2**), checkpoints under  
   `/storage/models/<abbr>_questioner_v1/`.
4. **`model_merger.py`** merges FSDP shards at  
   `.../global_step_<Q_MERGE>/actor`  
   with **`Q_MERGE = Q_STEPS - 1`** (default **1** → **`global_step_1`**).
5. vLLM reward processes are **killed** before the solver phase.

**Valid questioner path for later steps:**

`/storage/models/<abbr>_questioner_v1/global_step_<Q_MERGE>/actor/huggingface`

### STEP 2 / 2 — Solver v1

Requires **`HUGGINGFACENAME`** set (fail-fast if missing).

| Substep | What happens |
|--------|----------------|
| **2a** | [`question_generate/question_generate.bash`](../question_generate/question_generate.bash): **8** workers on GPUs **0–7** load the **merged questioner** in vLLM and write  
`/storage/generated_question/<solver_save>_<0..7>.json`  
(**`SMOKE_QUESTIONS_PER_GPU`** questions per file; default **8** → **64** total generations). **All workers must exit 0** before 2b. |
| **2b** | [`question_evaluate/evaluate.sh`](../question_evaluate/evaluate.sh): **8** workers read those JSONs, run majority-vote scoring with the **base model**, write  
`*_<i>_results.json`. |
| **2c** | [`question_evaluate/upload.py`](../question_evaluate/upload.py): merge results, filter scores (smoke uses **0.0–1.0**), push to **`{HUGGINGFACENAME}/{solver_save}`** with config name = experiment. **`--train_rows $N_TRAIN`** uploads **exactly `N_TRAIN`** rows (default **8**) so Hub train size matches rollout / `max_train_samples`. |
| **2d** | **Solver GRPO** on **8** GPUs; `data.train_files=${HUGGINGFACENAME}/${solver_save}@train`. |
| **2e** | Merge solver checkpoint at `global_step_<S_MERGE>`, **`S_MERGE = S_STEPS - 1`**. |

### STEP 3 — Optional GSM8K sanity check

[`evaluation/generate.py`](../evaluation/generate.py) on **GPU 0**; failures are **non-fatal** (`|| echo WARN`).

### After success

`modal_run.py` calls **`volume.commit()`** so new data is persisted on the **Modal volume**.

---

## Smoke scale defaults (aligned on 8)

| Variable | Default | Role |
|----------|---------|------|
| `SMOKE_MAX_TRAIN_SAMPLES` / `SMOKE_MAX_VAL_SAMPLES` | **8** | Caps for questioner (math12k) and solver val; matches batch multiples. |
| `SMOKE_ROLLOUT_BATCH_SIZE` / `SMOKE_VAL_BATCH_SIZE` / `SMOKE_GLOBAL_BATCH_SIZE` | **8** | Must be divisible by **4** (questioner) and **8** (solver) for even `DataProto` chunking. |
| `SMOKE_QUESTIONER_MAX_STEPS` | **2** | Questioner PPO steps; merge at **`global_step_(steps-1)`**. |
| `SMOKE_SOLVER_MAX_STEPS` | **4** | Solver PPO steps. |
| `SMOKE_QUESTIONS_PER_GPU` | **8** | **8×8 = 64** generated questions so evaluation/filtering usually still yields **≥ 8** rows before `--train_rows` truncation. |

Modal forwards these as **`SMOKE_*`** env vars from [`modal_run.py`](../modal_run.py) defaults.

---

## Storage: Modal volume vs Hugging Face

| Location | Contents |
|----------|----------|
| **Modal volume** (`/storage`) | `models/` (checkpoints), `generated_question/`, `temp_results/`, `evaluation/`, **`hf_cache/`** (HF model cache) |
| **Hugging Face Hub** | Solver **train** dataset pushed by `upload.py` (private repo under **`HUGGINGFACENAME`**); solver training reads **`{HUGGINGFACENAME}/<solver_save>@train`**. |

---

## WandB and logging

- **`WANDB_MODE=disabled`** in `smoke_test.sh` by default (fewer network/DNS issues on Modal).
- Smoke **`trainer.logger='[console]'`** for questioner and solver.

---

## Repository files (Modal / smoke–specific)

| Piece | Role |
|--------|------|
| [`modal_run.py`](../modal_run.py) | App, image, volume, `run_smoke_test`, env (`VLLM_DISABLE_COMPILE_CACHE`, `NO_PROXY`, **`TORCHINDUCTOR_MAX_AUTOTUNE=0`**, etc.) |
| [`scripts/smoke_test.sh`](../scripts/smoke_test.sh) | Full orchestration; eager vLLM + no actor `torch.compile` for smoke; clears **`/tmp/torchinductor_*`** |
| [`scripts/modal_run_smoke_detach.sh`](../scripts/modal_run_smoke_detach.sh) | `modal run --detach modal_run.py` |
| [`question_generate/question_generate.py`](../question_generate/question_generate.py) | Prepends **repo root** to `sys.path` before `import evaluation` |
| [`question_generate/question_generate.bash`](../question_generate/question_generate.bash) | `set -euo pipefail`; **wait each PID** so 2a failure stops before 2b |
| [`question_evaluate/evaluate.sh`](../question_evaluate/evaluate.sh) | Quoted args; **wait all 8** jobs; propagate non-zero exit |
| [`question_evaluate/upload.py`](../question_evaluate/upload.py) | **`--train_rows`**, smoke score band **0–1**, empty-filter checks |
| [`verl/trainer/data_loader.py`](../verl/trainer/data_loader.py) | **`drop_last`** only if `len(train) > rollout_batch_size` (avoids **0** train batches on tiny HF splits) |
| [`evaluation/generate.py`](../evaluation/generate.py) | Repo root on `sys.path`; **`STORAGE_PATH`** required |

---

## Findings and fixes (debugging history)

### 1. Empty solver train split / `SchemaInferenceError` (0 examples)

**Cause:** Full pipeline uses score band **0.3–0.8** in `upload.py`. Smoke majority-vote scores are often **0** or **1**, so **no rows** passed the filter → empty Hub dataset → `load_dataset` failed.

**Fix:** Smoke **2c** uses **`--min_score 0.0 --max_score 1.0`**. **`--train_rows "$N_TRAIN"`** uploads **exactly 8** rows when enough candidates exist.

### 2. Train dataloader `assert len(train_dataloader) >= 1`

**Cause:** **`drop_last=True`** with **5** train rows and **`rollout_batch_size=8`** → **0** batches.

**Fix:** [`data_loader.py`](../verl/trainer/data_loader.py): **`drop_last`** only when **`len(train_dataset) > rollout_batch_size`**.

### 3. Missing `generated_question/*.json` and “Input file not found” in evaluate

**Cause A:** **`ModuleNotFoundError: evaluation`** — running `python question_generate/question_generate.py` only adds **`question_generate/`** to `sys.path`, not the repo root.

**Fix:** Insert **repo root** on `sys.path` before importing `evaluation`.

**Cause B:** **`question_generate.bash`** used bare **`wait`**, which does **not** fail the script when background jobs crash → **2b ran with no files**.

**Fix:** Wait **each PID** and **`exit 1`** if any worker fails.

### 4. Count mismatch (e.g. 5 rows vs expected 8)

**Cause:** Evaluation drops many items (parse / grader / filters). **24** raw questions could shrink to **5**.

**Fix:** Default **`SMOKE_QUESTIONS_PER_GPU=8`** (64 total); **`--train_rows`** enforces **exact Hub train size = `N_TRAIN`**.

### 5. `BackendCompilerFailed` / `JSONDecodeError` (TorchInductor autotune cache)

**Cause:** vLLM compilation path hit **corrupt or bad TorchInductor autotune JSON** under **`/tmp/torchinductor_*`**.

**Fix (smoke):** **`worker.rollout.enforce_eager=true`**, **`worker.actor.use_torch_compile=false`** in `smoke_test.sh`; **`rm -rf /tmp/torchinductor_root`** at smoke start; **`TORCHINDUCTOR_MAX_AUTOTUNE=0`** in `modal_run.py` env.

### 6. STEP 3 GSM8K `import evaluation` failure

**Cause:** Same **`sys.path`** issue as question generation when running `python evaluation/generate.py`.

**Fix:** Repo root on `sys.path` at top of [`evaluation/generate.py`](../evaluation/generate.py).

### 7. Operational gotchas (retained from earlier work)

- **FSDP / attention:** [`verl/workers/fsdp_workers.py`](../verl/workers/fsdp_workers.py) falls back to **SDPA** if `flash_attn` is not installed.
- **Modal mounts:** `image.add_local_dir` + `FilePatternMatcher` (not deprecated `Mount.from_local_dir`).
- **Reward HTTP:** **`127.0.0.1`**, not **`0.0.0.0`**.
- **vLLM readiness:** **`GET /health`** on Flask wrapper ([`start_vllm_server.py`](../vllm_service_init/start_vllm_server.py)).
- **`caller_penalty.py`:** `extract_boxed_content` returns a **string** — do not index like a list.
- **DNS / long streams:** prior failures with **`nodename nor servname`** or flaky streams; **`WANDB_MODE=disabled`** in smoke reduces Hub-adjacent noise.
- **`HUGGINGFACENAME`:** must be set before solver phase (upload + `train_files`).

---

## GPU layout (8-GPU Modal worker)

- **GPUs 0–3:** Questioner GRPO.
- **GPUs 4–7:** Four vLLM instances (**5000–5003**) for **caller_penalty** during questioner training only; stopped before solver.
- **Solver:** training uses **all 8** GPUs per `trainer.n_gpus_per_node=8`.

---

## Security

Do **not** commit **`.env`**, **`tokens.json`**, or API keys. Rotate any key that was exposed.

---

## Quick reference

```bash
modal run --detach modal_run.py
modal app list
modal app logs r-zero-main
modal app stop <app_id>
```

---

*Last updated: Modal smoke workflow, start commands, storage, and consolidated debugging notes (train rows, imports, bash wait, Inductor/vLLM compile, dataloader `drop_last`).*
