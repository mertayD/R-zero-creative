#!/bin/bash
# =============================================================================
# creative_coevolve_smoke.sh  –  Co-evolving creative challenger + solver.
#
# Each iteration alternates two phases:
#
#   Phase A — Challenger training:
#     The *current solver* is frozen as the reward oracle (vLLM on GPU 7).
#     The challenger is trained (GPUs 0-3) with GRPO to generate prompts that
#     maximise the solver's uncertainty (R-Zero reward).
#
#   Phase B — Solver training:
#     The *updated challenger* generates writing prompts (GPU 7, then released).
#     The solver is trained (GPUs 0-3) with GRPO + normalised rank reward.
#
# Iteration i uses the checkpoints produced by iteration i-1.  Iteration 1
# starts both models from <base_model>.
#
# Usage:
#   bash scripts/creative_coevolve_smoke.sh \
#       <base_model> <Abbreviation> [<num_iters> [<num_train> [<num_val> [<seed>]]]]
#
#   base_model:   HF id or volume path — starting point for both models.
#   Abbreviation: short prefix for checkpoint directories.
#   num_iters:    co-evolution rounds (default 2).
#   num_train:    training rows per model per round (default 8).
#   num_val:      validation rows (default 2).
#   seed:         random seed (default 42).
#
# Scale controlled by env (set by train_creative_coevolve in modal_run.py):
#   SMOKE_CHALLENGER_MAX_STEPS  (default 4)
#   SMOKE_SOLVER_MAX_STEPS      (default 4)
#   SMOKE_SOLVER_ROLLOUT_N      (default 4)
# =============================================================================

set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-disabled}"
rm -rf /tmp/torchinductor_root /tmp/tinductor_* 2>/dev/null || true

base_model="$1"
Model_abbr="$2"
num_iters="${3:-2}"
num_train="${4:-8}"
num_val="${5:-2}"
seed="${6:-42}"

# ---------------------------------------------------------------------------
# Unique run ID
# ---------------------------------------------------------------------------
# Append a timestamp to Model_abbr so that every invocation creates its own
# checkpoint tree and never overwrites a previous run.
#
# SMOKE_RUN_ID is exported so creative_challenger_smoke.sh and
# creative_solver_smoke.sh know that Model_abbr already carries the stamp and
# must NOT add a second one.
# ---------------------------------------------------------------------------
RUN_TS=$(date +%Y%m%d_%H%M%S)
SMOKE_BASE_ABBR="$Model_abbr"         # original abbr, captured before timestamp
Model_abbr="${Model_abbr}_${RUN_TS}"
export SMOKE_RUN_ID="$RUN_TS"
export SMOKE_BASE_ABBR

# Read from env with defaults, then re-export so subscripts see the same values
C_STEPS="${SMOKE_CHALLENGER_MAX_STEPS:-4}"
S_STEPS="${SMOKE_SOLVER_MAX_STEPS:-4}"
export SMOKE_CHALLENGER_MAX_STEPS="$C_STEPS"
export SMOKE_SOLVER_MAX_STEPS="$S_STEPS"

C_MERGE=$((C_STEPS - 1))
S_MERGE=$((S_STEPS - 1))

echo "=========================================================="
echo " Creative Writing Co-Evolution Smoke"
echo " Base model  : $base_model"
echo " Abbreviation: $Model_abbr"
echo " Iterations  : $num_iters"
echo " Train rows  : $num_train  Val rows: $num_val  Seed: $seed"
echo " Storage     : $STORAGE_PATH"
echo " C_STEPS=$C_STEPS (merge @ step $C_MERGE)"
echo " S_STEPS=$S_STEPS (merge @ step $S_MERGE)"
echo "=========================================================="

# ---------------------------------------------------------------------------
# verify_checkpoint: confirm a merged HF checkpoint has actual model weights.
# Exits non-zero with a diagnostic if weights are absent.
# ---------------------------------------------------------------------------
verify_checkpoint() {
    local label="$1"
    local ckpt="$2"

    if [ ! -f "${ckpt}/config.json" ]; then
        echo ""
        echo "[ERROR] $label: config.json missing in merged checkpoint."
        echo "        Path: $ckpt"
        echo "        Did the training / merger script exit cleanly?"
        exit 1
    fi

    # Accept single-file or sharded safetensors, or legacy pytorch_model.bin
    if ls "${ckpt}"/model*.safetensors  2>/dev/null | grep -q . || \
       ls "${ckpt}"/pytorch_model*.bin  2>/dev/null | grep -q .; then
        echo "[OK] $label checkpoint verified (weights present): $ckpt"
        return 0
    fi

    echo ""
    echo "======================================================================"
    echo "[ERROR] $label checkpoint has NO model weight files!"
    echo "        Path : $ckpt"
    echo "        Files: $(ls "${ckpt}" 2>/dev/null | tr '\n' '  ')"
    echo ""
    echo "  model_merger.py ran but did not produce model.safetensors or"
    echo "  pytorch_model.bin.  Possible causes:"
    echo "    1. Merger ran out of CPU RAM (needs ~2× model size)."
    echo "    2. save_pretrained got a state_dict with mismatched keys."
    echo "    3. Merger exited non-zero (check script output above)."
    echo ""
    echo "  Tip: inspect the actor directory for .pt shards:"
    echo "    ls -lh ${ckpt}/../"
    echo "======================================================================"
    exit 1
}

# ---------------------------------------------------------------------------
# Iteration loop
# ---------------------------------------------------------------------------
# Start with base model for both roles
current_solver="$base_model"
current_challenger="$base_model"

for iter in $(seq 1 "$num_iters"); do
    iter_abbr="${Model_abbr}_iter${iter}"

    echo ""
    echo "=========================================================="
    echo " CO-EVOLUTION ITERATION  $iter / $num_iters  (run: $RUN_TS)"
    echo "   solver init     : $current_solver"
    echo "   challenger init : $current_challenger"
    echo "   (iter 1 = both start from base model; later iters use prior ckpts)"
    echo "=========================================================="

    # -----------------------------------------------------------------------
    # Phase A: Train challenger
    #   solver_model     = current_solver   → reward oracle on GPU 7
    #   challenger_model = current_challenger → model being trained on GPUs 0-3
    # -----------------------------------------------------------------------
    echo ""
    echo ">>> Phase A — Challenger training (iter $iter)"
    echo "    reward solver : $current_solver"
    echo "    init model    : $current_challenger"

    bash scripts/creative_challenger_smoke.sh \
        "$current_solver" \
        "$current_challenger" \
        "$iter_abbr" \
        "$num_train" "$num_val" "$seed"

    # Path where model_merger.py wrote the merged HF checkpoint
    new_challenger_ckpt="${STORAGE_PATH}/models/${iter_abbr}_challenger_v1/global_step_${C_MERGE}/actor/huggingface"
    verify_checkpoint "Challenger iter${iter}" "$new_challenger_ckpt"

    # -----------------------------------------------------------------------
    # Phase B: Train solver
    #   solver_model          = current_solver      → model being trained
    #   challenger_checkpoint = new_challenger_ckpt → prompt generator on GPU 7
    # -----------------------------------------------------------------------
    echo ""
    echo ">>> Phase B — Solver training (iter $iter)"
    echo "    init model        : $current_solver"
    echo "    challenger prompts: $new_challenger_ckpt"

    bash scripts/creative_solver_smoke.sh \
        "$current_solver" \
        "$new_challenger_ckpt" \
        "$iter_abbr" \
        "$num_train" "$num_val" "$seed"

    new_solver_ckpt="${STORAGE_PATH}/models/${iter_abbr}_solver_v1/global_step_${S_MERGE}/actor/huggingface"
    verify_checkpoint "Solver iter${iter}" "$new_solver_ckpt"

    # Advance both pointers for the next iteration
    current_challenger="$new_challenger_ckpt"
    current_solver="$new_solver_ckpt"

    echo ""
    echo "=== Iteration $iter complete ==="
    echo "    Challenger: $current_challenger"
    echo "    Solver    : $current_solver"
done

echo ""
echo "=========================================================="
echo " Creative Co-Evolution Smoke Complete"
echo " Iterations    : $num_iters"
echo " Final challenger: $current_challenger"
echo " Final solver    : $current_solver"
echo "=========================================================="
