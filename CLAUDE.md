# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

R-Zero is a self-supervised, co-evolutionary framework that teaches LLMs to improve reasoning without labeled data. A **Challenger (Questioner)** model generates hard problems; a **Solver** model improves by solving them. Each iteration, the questioner trains on the previous solver, creating a bootstrapped curriculum from a base model alone.

Paper: https://arxiv.org/abs/2508.05004

## Environment Setup

```bash
pip install -r requirements.txt

export STORAGE_PATH="/path/to/your/storage"
export HUGGINGFACENAME="yourhuggingfacename"

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" \
         "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

API keys go in `tokens.json` (HuggingFace + WandB). OpenAI key for `evaluation/results_recheck.py`. For Modal deployments, configure `.env` with `APP_NAME`, `VOLUME_NAME`, `MODAL_FULL_GPU`, `HUGGINGFACENAME`.

## Key Commands

**Full 5-iteration training:**
```bash
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

**Smoke test (one iteration, tiny data budget — validates the full pipeline):**
```bash
# On Modal (preferred):
modal run --detach modal_run.py
# or with overrides:
modal run --detach modal_run.py --base-model Qwen/Qwen3-4B-Base --smoke-max-train-samples 8

# Locally:
bash scripts/smoke_test.sh <Base_Model> <Abbreviation>
```

**Core trainer (used by all training scripts):**
```bash
python3 -m verl.trainer.main config=examples/config.yaml [overrides...]
```

**Merge FSDP checkpoint to HuggingFace format (required after each training step):**
```bash
python scripts/model_merger.py --local_dir $STORAGE_PATH/models/<name>/global_step_N/actor
```

**Evaluation on all benchmarks:**
```bash
bash evaluation/evaluate.bash <model_path>
# Single benchmark:
python evaluation/generate.py --model <model_path> --dataset gsm8k
```

**Question generation (after questioner trained and merged):**
```bash
bash question_generate/question_generate.bash <questioner_model_path> <questions_per_gpu> <experiment_name>
```

**Question evaluation + HuggingFace upload:**
```bash
bash question_evaluate/evaluate.sh <base_model> <experiment_name>
python question_evaluate/upload.py --repo_name <name> --min_score 0.3 --max_score 0.8 \
    --experiment_name <name> --train_rows <N>
```

## Architecture

### Training Loop (one iteration)

1. **Start vLLM reward servers** — `bash vllm_service_init/start.sh <base_model> <run_id>` launches 4 Flask-wrapped vLLM instances on ports 5000–5003 (GPUs 4–7). These score candidate answers for the questioner's reward signal.

2. **Train questioner** — `verl.trainer.main` with `reward_function=caller_penalty.py`. The reward penalizes duplicate questions (BLEU clustering) and rewards difficulty in the solver-score zone.

3. **Merge questioner checkpoint** — FSDP shards → single HuggingFace model via `model_merger.py`.

4. **Generate questions** — questioner model generates math problems across 8 GPUs.

5. **Evaluate questions** — base model answers each question 10× with majority vote; score = fraction correct.

6. **Upload dataset** — questions scoring 0.3–0.8 (medium difficulty) are uploaded to HuggingFace Hub as the solver's training set. Smoke test uses 0.0–1.0 since very few questions are generated.

7. **Train solver** — `verl.trainer.main` with `data.train_files` pointing to the Hub dataset.

8. **Merge solver checkpoint** — same as step 3.

### Core Components

| Path | Purpose |
|------|---------|
| `verl/trainer/main.py` | Entry point; reads OmegaConf config, starts `RayPPOTrainer` |
| `verl/trainer/ray_trainer.py` | Orchestrates GRPO training loop with Ray workers |
| `verl/trainer/data_loader.py` | HuggingFace dataset loading with drop_last logic |
| `verl/trainer/core_algos.py` | GRPO algorithm implementation |
| `verl/workers/fsdp_workers.py` | FSDP actor/critic workers |
| `verl/workers/rollout/` | vLLM-based rollout generation |
| `examples/reward_function/caller_penalty.py` | Questioner reward: calls vLLM servers, scores diversity + difficulty |
| `examples/reward_function/math.py` | Math answer correctness scoring |
| `vllm_service_init/start_vllm_server.py` | Flask + vLLM server used for reward computation |
| `question_generate/question_generate.py` | Generates questions from trained questioner |
| `question_evaluate/evaluate.py` | Majority-vote scoring of generated questions |
| `question_evaluate/upload.py` | Filters by difficulty band, uploads to HuggingFace Hub |
| `evaluation/generate.py` | Evaluates a model on benchmark datasets |
| `evaluation/datasets_loader.py` | Unified loader for MATH, GSM8K, AMC, Minerva, Olympiad, AIME, etc. |
| `scripts/model_merger.py` | Merges FSDP distributed shards into HuggingFace format |
| `modal_run.py` | Modal deployment: mounts repo + storage, sets env, calls smoke_test.sh |

### Config System

Training is configured via `examples/config.yaml` with OmegaConf dot-notation overrides on the CLI. Key sections: `data`, `worker.actor`, `worker.rollout`, `worker.reward`, `trainer`. The questioner and solver use the same trainer but different reward functions and data sources.

### GPU Layout (8× A100 reference)

- **Questioner training**: GPUs 0–3 (FSDP actor) + GPUs 4–7 (4 vLLM reward servers)
- **Solver training**: all 8 GPUs (FSDP actor)
- **Question generation**: all 8 GPUs (vLLM inference)
- **Evaluation**: GPU 0 only

### Data Flow

```
Base model
  → questioner training (GRPO + vLLM reward servers)
  → questioner checkpoint (merged)
  → question generation (vLLM)
  → question evaluation (majority vote, base model)
  → HuggingFace Hub dataset (difficulty-filtered)
  → solver training (GRPO on generated questions)
  → solver checkpoint (merged)
  → [next iteration: questioner trains on solver]
```
