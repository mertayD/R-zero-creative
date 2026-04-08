# Modal smoke experiment — context

This document describes how the **R-Zero end-to-end smoke run** was wired to **Modal**, what differs from the stock README/`main.sh` workflow, and operational notes. It is meant as **handoff context** for future runs and debugging.

---

## What this is (and is not)

- **Goal:** Run **one** full R-Zero iteration (questioner → merge → generate/evaluate/upload → solver → merge → optional GSM8K check) on **small data** to validate the pipeline before scaling.
- **Implementation:** A **new** script [`scripts/smoke_test.sh`](../scripts/smoke_test.sh) mirrors the **stages** of the paper/README loop, not a direct call to [`scripts/main.sh`](../scripts/main.sh). `main.sh` runs **five** iterations; the smoke path runs **one**.
- **Upstream:** The original open-source repo does **not** ship Modal code. [`modal_run.py`](../modal_run.py) is **project-local** glue: image, volume, secrets, and a single GPU function that invokes `smoke_test.sh`.

---

## Repository additions and touched files

| Piece | Role |
|--------|------|
| [`modal_run.py`](../modal_run.py) | Modal `App`, container image, `run_smoke_test` function, local entrypoint |
| [`scripts/smoke_test.sh`](../scripts/smoke_test.sh) | Bash orchestration for one iteration + env-driven scale |
| [`scripts/modal_run_smoke_detach.sh`](../scripts/modal_run_smoke_detach.sh) | Thin wrapper: `modal run --detach modal_run.py "$@"` |
| [`verl/trainer/config.py`](../verl/trainer/config.py) | `max_train_samples` / `max_val_samples` on `DataConfig` |
| [`verl/utils/dataset.py`](../verl/utils/dataset.py) | Optional cap after filtering; safer `format_prompt` guards |
| [`verl/trainer/data_loader.py`](../verl/trainer/data_loader.py) | Passes sample caps into `RLHFDataset` |
| [`examples/reward_function/caller_penalty.py`](../examples/reward_function/caller_penalty.py) | Reward HTTP to **127.0.0.1**; fix for `extract_boxed_content` (string vs list) |
| [`examples/reward_function/caller.py`](../examples/reward_function/caller.py) | Same **127.0.0.1** client URL as `caller_penalty.py` |
| [`vllm_service_init/start_vllm_server.py`](../vllm_service_init/start_vllm_server.py) | **`GET /health`** for readiness probes |
| [`verl/workers/fsdp_workers.py`](../verl/workers/fsdp_workers.py) | `flash_attn` optional → `sdpa` fallback |
| [`requirements.txt`](../requirements.txt) | `stopit` (used by scripts) |

---

## Prerequisites

1. **Modal:** CLI logged in; workspace/profile used for this project (e.g. `columbia-daplab` via `MODAL_PROFILE` in `.env`).
2. **`.env`** (gitignored): volume name, GPU string, paths, `HUGGINGFACENAME`, timeouts, **no need to commit secrets**. [`modal_run.py`](../modal_run.py) uses `python-dotenv` at import time **locally**.
3. **`tokens.json`** (local, gitignored recommended): `huggingface` and `wandb` keys. Read **only when `modal.is_local()`** so container import does not require `tokens.json` in `/root`.
4. **Modal secret:** At runtime the container expects **`HF_TOKEN`**, **`WANDB_API_KEY`**, **`HUGGINGFACENAME`**, **`STORAGE_PATH`** via `modal.Secret.from_dict` built from local `tokens.json` + `.env` when you deploy/run from your machine. Align secret name with your Modal dashboard if you use a named secret instead.
5. **Volume:** `modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)` — checkpoints, generated questions, HF cache, temp results live under `REMOTE_STORAGE_PATH` (default `/storage`).

---

## How the Modal run works

1. **Image:** `nvidia/cuda:12.6.0-devel-ubuntu22.04` + Python 3.10; PyTorch cu126 wheels; `vllm==0.9.1`; optional `flash-attn` after `pip install wheel`; extra pip deps including `ray`, `wandb`, `mathruler`, `codetiming`, etc. Repo mounted with `add_local_dir(..., copy=False)` and ignore rules for `.git`, `__pycache__`, `.env`.
2. **Function:** Single `@app.function` with **`gpu=MODAL_FULL_GPU`** (default **`A100-40GB:8`**), attached volume at **`/storage`**, long **`timeout`**, training-friendly `env` (including **`NO_PROXY`/`no_proxy`** for local vLLM HTTP).
3. **Inside the container:** `cd` to `REMOTE_REPO_PATH`, ensure dirs under `$STORAGE_PATH`, set **`HF_HOME`** / **`HUGGINGFACE_HUB_CACHE`** to **`/storage/hf_cache`** so model downloads persist on the volume, then:

   ```bash
   bash scripts/smoke_test.sh <base_model> <abbr>
   ```

4. **After success:** `volume.commit()` writes new artifacts to the persistent volume.

---

## Smoke scale defaults (why 8 rows and batch 8)

- **`SMOKE_MAX_TRAIN_SAMPLES` / `SMOKE_MAX_VAL_SAMPLES`:** default **8** — small, and matches **`rollout_batch_size=8`** with `drop_last=True` in the dataloader (one full batch per epoch from the capped set).
- **`SMOKE_ROLLOUT_BATCH_SIZE` (and val / global):** default **8** — **required** for verl’s `DataProto.chunk(chunks=world_size)`:
  - Questioner uses **`trainer.n_gpus_per_node=4`** → batch must be divisible by **4**.
  - Solver uses **`trainer.n_gpus_per_node=8`** → batch must be divisible by **8**.
  - Using **8** satisfies both. A value like **2** fails with: `AssertionError: only support equal chunk. Got size of DataProto 2 and chunk 4.`

Modal CLI parameters on [`modal_run.py`](../modal_run.py) (`smoke_*`) are forwarded as **`SMOKE_*`** environment variables for `smoke_test.sh`.

---

## GPU layout on the 8-GPU worker

- **GPUs 0–3:** Questioner GRPO (and later solver training uses all 8 per config).
- **GPUs 4–7:** Four **vLLM** processes (ports **5000–5003**) for the **caller_penalty** reward during questioner training only; stopped before solver phase.

---

## Running (always detach for long jobs)

```bash
modal run --detach modal_run.py
```

Or with overrides:

```bash
modal run --detach modal_run.py \
  --base-model Qwen/Qwen3-4B-Base \
  --abbr qwen3-4b-smoke \
  --smoke-max-train-samples 8 \
  --smoke-questioner-max-steps 2 \
  --smoke-solver-max-steps 4
```

Wrapper:

```bash
bash scripts/modal_run_smoke_detach.sh
```

**Why `--detach`:** Without it, the app is **ephemeral**; losing the local CLI (terminal close, VPN drop) can **stop** the remote job. Detached apps keep running; follow with the **Modal dashboard** or `modal app logs <app-id>`.

**Resume:** Modal does not “resume” a **stopped** app ID. Training **checkpoint resume** would require passing **`load_checkpoint_path`** (or equivalent) into `verl` — not implemented in the smoke script today.

---

## Storage: Modal volume vs Hugging Face

| Location | Contents |
|----------|----------|
| **Modal volume** (`/storage`) | Checkpoints under `models/`, `generated_question/`, `temp_results/`, `evaluation/`, **`hf_cache/`** (HF model hub cache) |
| **Hugging Face Hub** | Filtered solver dataset from **`question_evaluate/upload.py`** (repo name derived from experiment); base model IDs still pulled from Hub into the volume cache |

---

## WandB and logging

- Smoke sets **`WANDB_MODE=disabled`** by default in `smoke_test.sh` to avoid extra network and logging noise.
- Trainer uses **`trainer.logger='[console]'`** in smoke for the same reason. You can re-enable WandB by changing env/config if needed.

---

## Fixes and gotchas encountered

1. **Modal API:** Use **`image.add_local_dir`** + **`FilePatternMatcher`**, not deprecated `Mount.from_local_dir` / `mounts=`.
2. **`tokens.json` in container:** Only read on **local** import; container relies on **secret env vars**.
3. **Reward HTTP:** Use **`http://127.0.0.1:<port>`** instead of **`0.0.0.0`** for `requests` (resolution / proxy issues).
4. **`/health`:** Added on Flask vLLM wrapper; smoke polls it instead of misusing `/hello`.
5. **`caller_penalty.py`:** `extract_boxed_content` returns a **string** — do not index like a list.
6. **`verl/utils/dataset.py`:** Guard **`self.format_prompt`** before substring checks.
7. **`fsdp_workers.py`:** **`flash_attention_2`** only if `flash_attn` imports; else **`sdpa`**.
8. **Dependencies:** `stopit`, `codetiming`, **`wheel`** before building flash-attn in the image.
9. **Detached + streaming:** Local `modal run` may still stream until the function finishes; closing the terminal is safe **only with `--detach`**.

---

## Security note

Do **not** commit **`.env`**, **`tokens.json`**, or real API keys. This doc intentionally avoids pasting secrets. Rotate any key that was ever committed or shared in chat.

---

## Quick reference commands

```bash
# List apps
modal app list

# Logs for a detached run
modal app logs <APP_ID>

# Stop a detached app
modal app stop <APP_ID>
```

---

*Last updated to match the smoke/Modal setup in this repo (Modal detached runs, 8-sample cap, batch size 8, single iteration).*
