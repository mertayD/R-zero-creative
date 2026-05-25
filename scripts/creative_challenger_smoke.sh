#!/bin/bash
# =============================================================================
# creative_challenger_smoke.sh  –  Train the creative writing challenger.
#
# Usage:
#   bash scripts/creative_challenger_smoke.sh \
#       <solver_model> <challenger_model> <Abbreviation> <num_train> <num_val> <seed>
#
#   solver_model:     HF id or volume path — loaded on GPU 7 as the vLLM reward server.
#   challenger_model: HF id or volume path — trained by VERL on GPUs 0-3.
#   Abbreviation:     short name for checkpoint directories.
#
# Training data: domain/subdomain pairs from DomainSampler (creative_writing_prompts.py),
#   formatted as full one-shot prompts via build_one_shot_prompt().
# Reward (live): challenger generates a prompt → solver answers → BatchEvalAgent scores
#   → R-Zero uncertainty reward returned by creative_writing_caller.py.
#
# Scale controlled by env (set by train_creative_challenger in modal_run.py):
#   SMOKE_CHALLENGER_MAX_STEPS  (default 4)
# =============================================================================

set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-disabled}"
rm -rf /tmp/torchinductor_root /tmp/tinductor_* 2>/dev/null || true

solver_model="$1"
challenger_model="$2"
Model_abbr="$3"
num_train="${4:-8}"
num_val="${5:-2}"
seed="${6:-42}"

C_STEPS="${SMOKE_CHALLENGER_MAX_STEPS:-4}"
C_MERGE=$((C_STEPS - 1))
SAVE_NAME="${Model_abbr}_challenger_v1"

# Cap rollout/actor batch sizes to num_train so VERL always has enough data
ROLLOUT_BATCH=$(( num_train < 8 ? num_train : 8 ))
ACTOR_GLOBAL_BATCH=$(( ROLLOUT_BATCH < 8 ? ROLLOUT_BATCH : 8 ))

echo "=========================================================="
echo " Creative Writing Challenger Smoke Training"
echo " Solver model    : $solver_model"
echo " Challenger model: $challenger_model"
echo " Abbreviation    : $Model_abbr"
echo " Train rows      : $num_train  Val rows: $num_val  Seed: $seed"
echo " Storage         : $STORAGE_PATH"
echo " Max steps       : $C_STEPS  (merge at step $C_MERGE)"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Step 0: Build train/val parquet from domain/subdomain pairs
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 0 — Building VERL parquet from WritingBench domains"

python3 - <<PYEOF
import sys, os
sys.path.insert(0, os.environ.get("REMOTE_REPO_PATH", "/root/R-Zero"))

from pathlib import Path
import pandas as pd
from question_generate.one_shot_creative_question_generate import DomainSampler, build_one_shot_prompt

sampler = DomainSampler(seed=int("$seed"))
out_dir = Path("${STORAGE_PATH}/creative_smoke")
out_dir.mkdir(parents=True, exist_ok=True)

def make_row():
    domain_key, subdomain = sampler.sample_domain_subdomain_pair()
    _, user_prompt, _     = build_one_shot_prompt(domain_key, subdomain)
    return {"problem": user_prompt, "answer": ""}

train_rows = [make_row() for _ in range(int("$num_train"))]
val_rows   = [make_row() for _ in range(int("$num_val"))]

pd.DataFrame(train_rows).to_parquet(out_dir / "${Model_abbr}_train.parquet", index=False)
pd.DataFrame(val_rows).to_parquet(out_dir   / "${Model_abbr}_val.parquet",   index=False)
print(f"[OK] train={len(train_rows)}  val={len(val_rows)}  -> {out_dir}")
PYEOF

echo "Parquet conversion done."

# ---------------------------------------------------------------------------
# Step 1: Start vLLM solver server on GPU 7 (standard OpenAI-compatible API)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 1 — Starting vLLM solver server on GPU 7 (port 5000): $solver_model"

CUDA_VISIBLE_DEVICES=7 python3 -m vllm.entrypoints.openai.api_server \
    --model "$solver_model" \
    --port 5000 \
    --dtype auto \
    --disable-log-requests &

VLLM_PID=$!
echo "vLLM solver PID=$VLLM_PID"

# Poll /health until ready (up to 10 min)
RETRIES=0
until curl -sf "http://127.0.0.1:5000/health" > /dev/null 2>&1; do
    sleep 10
    RETRIES=$((RETRIES + 1))
    if [ $RETRIES -ge 60 ]; then
        echo "[ERROR] vLLM solver not ready after 10 min"
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
    echo "  ... waiting for vLLM solver (${RETRIES}/60)"
done
echo "vLLM solver server ready."

# ---------------------------------------------------------------------------
# Step 2: Train challenger with VERL GRPO
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 2 — Training challenger ($challenger_model, $C_STEPS steps)"

TRAIN_PARQUET="${STORAGE_PATH}/creative_smoke/${Model_abbr}_train.parquet"
VAL_PARQUET="${STORAGE_PATH}/creative_smoke/${Model_abbr}_val.parquet"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.rollout_batch_size="$ROLLOUT_BATCH" \
    data.val_batch_size="$num_val" \
    worker.actor.model.model_path="$challenger_model" \
    trainer.experiment_name="$SAVE_NAME" \
    trainer.save_checkpoint_path="${STORAGE_PATH}/models/${SAVE_NAME}" \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/creative_writing_caller.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    worker.rollout.n=2 \
    worker.rollout.enforce_eager=true \
    worker.actor.use_torch_compile=false \
    worker.actor.global_batch_size="$ACTOR_GLOBAL_BATCH" \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.max_steps="$C_STEPS" \
    trainer.save_freq=1 \
    trainer.logger='[console]'

echo "Challenger training done — merging checkpoint at global_step_${C_MERGE}"
sleep 5

# ---------------------------------------------------------------------------
# Step 3: Merge checkpoint
# ---------------------------------------------------------------------------
python scripts/model_merger.py \
    --local_dir "${STORAGE_PATH}/models/${SAVE_NAME}/global_step_${C_MERGE}/actor"

echo "Checkpoint merged."

# ---------------------------------------------------------------------------
# Step 4: Kill vLLM server
# ---------------------------------------------------------------------------
sleep 5
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "vLLM solver stopped."

echo ""
echo "=========================================================="
echo " Creative Challenger Smoke Training Complete"
echo " Checkpoint: ${STORAGE_PATH}/models/${SAVE_NAME}/"
echo "=========================================================="
