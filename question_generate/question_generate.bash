set -euo pipefail
# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
export VLLM_DISABLE_COMPILE_CACHE=1

# Run 8 workers; require every exit 0. Plain `wait` does not fail the script if a
# background job crashes, so smoke/evaluate would run with missing *.json files.
pids=()
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python question_generate/question_generate.py \
    --model "$model_name" --suffix "$i" --num_samples "$num_samples" --save_name "$save_name" &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit $status
