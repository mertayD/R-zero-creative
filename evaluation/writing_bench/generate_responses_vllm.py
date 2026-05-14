"""
Batch response generator for WritingBench using vLLM.

Replaces upstream's generate_response.py (which calls a single-threaded `writer`
function over an OpenAI-compatible API). For local models we want batched
generation, so this driver loads the model once, prompts vLLM with the entire
query set in one call, and writes the official JSONL output schema:

    {"index": <int>, "response": "<text>"}

Sampling parameters are kept exactly aligned with the WritingBench leaderboard
(see https://github.com/X-PLUG/WritingBench, "Quick Start" section):
    top_p=0.8, top_k=20, temperature=0.7, max_length=16000

Resumable: if --output_file already exists, indices already in it are skipped.

Usage:
    python evaluation/writing_bench/generate_responses_vllm.py \
        --model Qwen/Qwen3-4B-Base \
        --query_file evaluation/writing_bench/subsets/smoke50.jsonl \
        --output_file $STORAGE_PATH/writing_bench/responses/qwen3-4b-base/smoke50.jsonl
"""

import argparse
import json
import os
import sys

# Make the bare `prompt` import inside upstream files resolvable when running
# this script from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import vllm
from transformers import AutoTokenizer


# Official WritingBench leaderboard generation params.
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 16000


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def build_prompt(tokenizer, query):
    """
    WritingBench provides the full instruction in `query`. Per the upstream
    generate_response.py, the writer is called with just the query as a single
    user turn; no system prompt is added. We mirror that exactly.
    """
    chat = [{"role": "user", "content": query}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True
        )
    # Fallback for raw base models without a chat template — just the query.
    return query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HF id or local path of the model under evaluation.")
    parser.add_argument("--query_file", type=str, required=True,
                        help="JSONL with WritingBench fields: index, query, ...")
    parser.add_argument("--output_file", type=str, required=True,
                        help="JSONL where {index, response} records are written.")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Override vLLM max_model_len. Useful for context-limited models.")
    args = parser.parse_args()

    queries = load_jsonl(args.query_file)
    if not queries:
        raise SystemExit(f"No queries found in {args.query_file}")

    # Resume: skip indices already present in output_file.
    done = {r["index"] for r in load_jsonl(args.output_file)}
    todo = [q for q in queries if q["index"] not in done]
    print(f"[generate] queries={len(queries)}  done={len(done)}  todo={len(todo)}")
    if not todo:
        print("[generate] Nothing to do — output already complete.")
        return

    print(f"[generate] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm_kwargs = dict(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    model = vllm.LLM(**llm_kwargs)

    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    prompts = [build_prompt(tokenizer, q["query"]) for q in todo]
    print(f"[generate] Generating {len(prompts)} responses with "
          f"top_p={args.top_p}, top_k={args.top_k}, temp={args.temperature}, "
          f"max_tokens={args.max_tokens}")
    outputs = model.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # vLLM preserves input order, so zip todo with outputs is safe.
    records = [
        {"index": q["index"], "response": out.outputs[0].text}
        for q, out in zip(todo, outputs)
    ]
    append_jsonl(records, args.output_file)
    print(f"[generate] Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
