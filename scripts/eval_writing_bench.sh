#!/usr/bin/env bash
# Run the full WritingBench eval pipeline on a single model.
#
# Stages:
#   1. Generate responses with vLLM using the official leaderboard sampling
#      params (top_p=0.8, top_k=20, temperature=0.7, max_tokens=16000).
#   2. Score each (response, criterion) pair with Claude-Sonnet-4-5 — the
#      judge currently used by the WritingBench leaderboard. Set
#      ANTHROPIC_API_KEY in your environment.
#   3. Aggregate scores into an .xlsx report.
#
# Usage:
#   bash scripts/eval_writing_bench.sh <model_name_or_path> <subset> <storage_path>
#
#   model_name_or_path : HF id or local path of the model under evaluation.
#   subset             : smoke50 | mid100 | full
#   storage_path       : root directory for outputs. Run artifacts land at
#                        <storage_path>/writing_bench/<model_slug>/<subset>/.
#                        If omitted, falls back to the STORAGE_PATH env var.
#                        Recommended to point at a folder OUTSIDE this repo
#                        (e.g. ~/wb-runs or a scratch disk) to avoid bloat.
#
# Required env:
#   ANTHROPIC_API_KEY  : Anthropic key for the Claude-Sonnet-4-5 judge.
#
# Examples:
#   bash scripts/eval_writing_bench.sh Qwen/Qwen3-4B-Base smoke50 ~/wb-runs
#   bash scripts/eval_writing_bench.sh Qwen/Qwen3-4B-Base mid100  /data/wb
#   bash scripts/eval_writing_bench.sh "$STORAGE_PATH/models/qwen3-4b-iter1" smoke50 "$STORAGE_PATH"

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_name_or_path> <smoke50|mid100|full> <storage_path>" >&2
    exit 1
fi

MODEL="$1"
SUBSET="$2"
STORAGE_PATH="${3:-${STORAGE_PATH:-}}"

if [[ -z "${STORAGE_PATH}" ]]; then
    echo "Error: storage_path is required (pass as 3rd arg or export STORAGE_PATH)." >&2
    exit 1
fi
mkdir -p "${STORAGE_PATH}"

: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set for the Claude-Sonnet-4-5 judge}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WB_DIR="${REPO_ROOT}/evaluation/writing_bench"
BENCH_ALL="${WB_DIR}/benchmark_query/benchmark_all.jsonl"
REQUIREMENT_DIR="${WB_DIR}/benchmark_query/requirement"

# Pick subset query file.
case "${SUBSET}" in
    smoke50) QUERY_FILE="${WB_DIR}/subsets/smoke50.jsonl" ;;
    mid100)  QUERY_FILE="${WB_DIR}/subsets/mid100.jsonl"  ;;
    full)    QUERY_FILE="${BENCH_ALL}" ;;
    *)
        echo "Unknown subset: ${SUBSET} (expected smoke50|mid100|full)" >&2
        exit 1
        ;;
esac

# Ensure the upstream data is present (needed for `full` and for
# calculate_scores.py's requirement-dimension breakdowns regardless of subset).
if [[ ! -f "${BENCH_ALL}" ]]; then
    echo "==> Upstream WritingBench data missing — fetching..."
    bash "${WB_DIR}/fetch_data.sh"
fi

# Slug the model name for output paths. Replace path separators and whitespace
# with underscores. Using POSIX bracket classes (`[[:space:]]`) instead of `\s`
# because `\s` is non-portable across sed implementations and was silently
# matching the literal letter `s` on some builds.
MODEL_SLUG="$(echo "${MODEL}" | sed -e 's|/|_|g' -e 's|[[:space:]]|_|g')"
RUN_DIR="${STORAGE_PATH}/writing_bench/${MODEL_SLUG}/${SUBSET}"
RESP_FILE="${RUN_DIR}/responses.jsonl"
SCORE_DIR="${RUN_DIR}/scores"
SCORE_FILE="${SCORE_DIR}/${MODEL_SLUG}.jsonl"
EXCEL_FILE="${RUN_DIR}/scores.xlsx"

mkdir -p "${RUN_DIR}" "${SCORE_DIR}"

echo "==> [1/3] Generating responses for ${MODEL} on ${SUBSET}"
python3 "${WB_DIR}/generate_responses_vllm.py" \
    --model "${MODEL}" \
    --query_file "${QUERY_FILE}" \
    --output_file "${RESP_FILE}"

echo "==> [2/3] Scoring with Claude-Sonnet-4-5"
# evaluate_benchmark.py imports `prompt` and `evaluator` as top-level packages,
# so we cd into WB_DIR for the call.
(
    cd "${WB_DIR}"
    python evaluate_benchmark.py \
        --evaluator claude \
        --query_criteria_file "${QUERY_FILE}" \
        --input_file "${RESP_FILE}" \
        --output_file "${SCORE_FILE}"
)

echo "==> [3/3] Aggregating scores -> ${EXCEL_FILE}"
python "${WB_DIR}/calculate_scores.py" \
    --score_dir "${SCORE_DIR}" \
    --benchmark_file "${BENCH_ALL}" \
    --output_excel "${EXCEL_FILE}" \
    --requirement_dir "${REQUIREMENT_DIR}"

echo "==> Done."
echo "  Responses : ${RESP_FILE}"
echo "  Scores    : ${SCORE_FILE}"
echo "  Report    : ${EXCEL_FILE}"
