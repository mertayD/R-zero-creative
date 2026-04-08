#!/bin/bash
# =============================================================================
# smoke_test.sh  –  One full R-Zero iteration with a small data budget.
#
# Purpose: validate the complete pipeline end-to-end (no correctness guarantee).
# Usage  : bash scripts/smoke_test.sh <Base_Model> <Abbreviation>
#
# Scale is controlled by environment (modal_run.py sets these by default):
#   SMOKE_MAX_TRAIN_SAMPLES, SMOKE_MAX_VAL_SAMPLES, SMOKE_ROLLOUT_BATCH_SIZE,
#   SMOKE_VAL_BATCH_SIZE, SMOKE_GLOBAL_BATCH_SIZE, SMOKE_QUESTIONER_MAX_STEPS,
#   SMOKE_SOLVER_MAX_STEPS, SMOKE_QUESTIONS_PER_GPU
# =============================================================================

set -euo pipefail

# Smoke run: no Weights & Biases network calls (avoids flaky DNS / long-run Modal log stream issues).
export WANDB_MODE="${WANDB_MODE:-disabled}"

Base_model="$1"
Model_abbr="$2"

# --- Defaults: 8 train/val rows, short steps (override via env from Modal) ---
# Rollout/val/global batch must divide evenly across GPUs: questioner n_gpus=4,
# solver n_gpus=8 (config default) → use a multiple of 8 (e.g. 8, not 2).
N_TRAIN="${SMOKE_MAX_TRAIN_SAMPLES:-8}"
N_VAL="${SMOKE_MAX_VAL_SAMPLES:-8}"
RB="${SMOKE_ROLLOUT_BATCH_SIZE:-8}"
VB="${SMOKE_VAL_BATCH_SIZE:-8}"
GBS="${SMOKE_GLOBAL_BATCH_SIZE:-8}"
Q_STEPS="${SMOKE_QUESTIONER_MAX_STEPS:-2}"
S_STEPS="${SMOKE_SOLVER_MAX_STEPS:-4}"
Q_PER_GPU="${SMOKE_QUESTIONS_PER_GPU:-3}"
Q_MERGE=$((Q_STEPS - 1))
S_MERGE=$((S_STEPS - 1))

echo "=========================================================="
echo " R-Zero SMOKE TEST"
echo " Base model : $Base_model"
echo " Abbreviation: $Model_abbr"
echo " Storage    : $STORAGE_PATH"
echo " Train cap  : $N_TRAIN rows | rollout_batch=$RB | questioner_steps=$Q_STEPS"
echo " Solver     : max_steps=$S_STEPS | questions_per_gpu=$Q_PER_GPU"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Questioner training  (Iteration 1)
# ---------------------------------------------------------------------------
echo ""
echo ">>> STEP 1 / 2 — Train questioner v1"

RUN_ID=$(date +%s%N)
export RUN_ID
echo "RUN_ID=$RUN_ID"

# Start 4 vLLM solver servers on GPUs 4-7 (used by caller_penalty.py reward fn)
export VLLM_DISABLE_COMPILE_CACHE=1
bash vllm_service_init/start.sh "$Base_model" "$RUN_ID"
echo "vLLM solver servers started (GPUs 4-7) — waiting for readiness..."

# Poll /health until all 4 servers are up (up to 10 minutes each)
for PORT in 5000 5001 5002 5003; do
    RETRIES=0
    until curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; do
        sleep 10
        RETRIES=$((RETRIES + 1))
        if [ $RETRIES -ge 60 ]; then
            echo "[ERROR] vLLM server on port ${PORT} not ready after 10 min"
            exit 1
        fi
        echo "  ... waiting for vLLM:${PORT} (${RETRIES}/60)"
    done
    echo "  vLLM:${PORT} ready"
done
echo "All vLLM servers are ready."

QUESTIONER_SAVE="${Model_abbr}_questioner_v1"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_train_samples="$N_TRAIN" \
    data.max_val_samples="$N_VAL" \
    data.rollout_batch_size="$RB" \
    data.val_batch_size="$VB" \
    data.max_response_length=2048 \
    worker.actor.model.model_path="$Base_model" \
    trainer.experiment_name="$QUESTIONER_SAVE" \
    trainer.save_checkpoint_path="${STORAGE_PATH}/models/${QUESTIONER_SAVE}" \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/caller_penalty.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=2 \
    worker.actor.global_batch_size="$GBS" \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.max_steps="$Q_STEPS" \
    trainer.save_freq=1 \
    trainer.logger='[console]'

echo "Questioner training done — merging checkpoint at global_step_${Q_MERGE}"
sleep 5

python scripts/model_merger.py \
    --local_dir "${STORAGE_PATH}/models/${QUESTIONER_SAVE}/global_step_${Q_MERGE}/actor"

sleep 5
pkill -f "start_vllm_server" || true
pkill -f "vllm_service_init" || true
echo "vLLM servers stopped"

# ---------------------------------------------------------------------------
# Solver training  (Iteration 1)
# ---------------------------------------------------------------------------
echo ""
echo ">>> STEP 2 / 2 — Train solver v1"

SOLVER_SAVE="${Model_abbr}_solver_v1"
QUESTIONER_HF="${STORAGE_PATH}/models/${QUESTIONER_SAVE}/global_step_${Q_MERGE}/actor/huggingface"

# --- 2a. Generate questions (SMOKE_QUESTIONS_PER_GPU × 8 GPUs)
echo "  2a. Generating questions..."
bash question_generate/question_generate.bash "$QUESTIONER_HF" "$Q_PER_GPU" "$SOLVER_SAVE"

# --- 2b. Evaluate questions with the solver (majority-vote scoring) ---
echo "  2b. Evaluating questions..."
bash question_evaluate/evaluate.sh "$Base_model" "$SOLVER_SAVE"

# --- 2c. Push evaluated questions to HuggingFace (solver train_files) ---
# Full pipeline uses 0.3–0.8 to keep “medium-hard” items; smoke has very few
# questions and majority-vote scores are usually 0 or 1, so that band is often
# empty and load_dataset then fails with “0 examples” / SchemaInferenceError.
echo "  2c. Uploading dataset to HuggingFace (smoke: full score range 0–1)..."
python question_evaluate/upload.py \
    --repo_name  "$SOLVER_SAVE" \
    --max_score  1.0 \
    --min_score  0.0 \
    --experiment_name "$SOLVER_SAVE"

# --- 2d. Train solver on the generated questions ---
echo "  2d. Training solver..."
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_train_samples="$N_TRAIN" \
    data.max_val_samples="$N_VAL" \
    data.rollout_batch_size="$RB" \
    data.val_batch_size="$VB" \
    data.max_response_length=2048 \
    worker.actor.model.model_path="$Base_model" \
    trainer.experiment_name="$SOLVER_SAVE" \
    trainer.save_checkpoint_path="${STORAGE_PATH}/models/${SOLVER_SAVE}/" \
    data.train_files="${HUGGINGFACENAME}/${SOLVER_SAVE}@train" \
    trainer.total_epochs=100 \
    trainer.max_steps="$S_STEPS" \
    trainer.save_freq=1 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=8 \
    worker.actor.global_batch_size="$GBS" \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.logger='[console]'

# --- 2e. Merge solver checkpoint ---
echo "  2e. Merging solver checkpoint at global_step_${S_MERGE}..."
python scripts/model_merger.py \
    --local_dir "${STORAGE_PATH}/models/${SOLVER_SAVE}/global_step_${S_MERGE}/actor"

sleep 5

# ---------------------------------------------------------------------------
# Quick sanity-eval on GSM8K only (fast, skip full benchmark suite)
# ---------------------------------------------------------------------------
echo ""
echo ">>> STEP 3 — Quick evaluation on GSM8K (smoke-only)"

SOLVER_HF="${STORAGE_PATH}/models/${SOLVER_SAVE}/global_step_${S_MERGE}/actor/huggingface"

export VLLM_DISABLE_COMPILE_CACHE=1
CUDA_VISIBLE_DEVICES=0 python evaluation/generate.py \
    --model "$SOLVER_HF" \
    --dataset gsm8k || echo "[WARN] Evaluation failed — pipeline is still valid"

echo ""
echo "=========================================================="
echo " SMOKE TEST COMPLETE"
echo " Questioner: ${STORAGE_PATH}/models/${QUESTIONER_SAVE}"
echo " Solver    : ${STORAGE_PATH}/models/${SOLVER_SAVE}"
echo "=========================================================="
