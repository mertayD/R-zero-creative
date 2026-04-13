#!/bin/bash
set -euo pipefail

model_name=$1
save_name=$2

pids=()
for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=$i python question_evaluate/evaluate.py \
    --model "$model_name" --suffix "$i" --save_name "$save_name" &
  pids[$i]=$!
done

timeout_duration=3600
(
  sleep "$timeout_duration"
  echo "Timeout reached. Killing remaining evaluate tasks..."
  for i in {0..7}; do
    if kill -0 "${pids[$i]}" 2>/dev/null; then
      kill -9 "${pids[$i]}" 2>/dev/null || true
      echo "Killed task $i"
    fi
  done
) &
_timeout_pid=$!

status=0
for i in {0..7}; do
  if ! wait "${pids[$i]}"; then
    status=1
  fi
done
kill "$_timeout_pid" 2>/dev/null || true
wait "$_timeout_pid" 2>/dev/null || true

exit $status
