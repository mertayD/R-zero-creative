"""
Batch response generator for WritingBench using the Tinker sampling API.

Sibling of generate_responses_vllm.py for when no local GPU is available:
identical prompt construction (single user turn, chat template when the
tokenizer has one), identical official leaderboard sampling params
(top_p=0.8, top_k=20, temperature=0.7, max_tokens=16000), identical output
schema ({"index": <int>, "response": "<text>"}) and resume semantics. Only
the tokenizer runs locally; token generation happens on Tinker's servers.

One deliberate divergence from the vLLM driver: <|im_end|> joins the stop set
next to eos. A base model driven through a chat template may only stop on
<|endoftext|>, and Tinker bills per generated token, so rambling to the 16k
cap is real money rather than idle GPU time.

Auth: the Tinker SDK reads TINKER_API_KEY (or the credential stored by
`tinker auth login`). Usage:
    python evaluation/writing_bench/generate_responses_tinker.py \
        --model Qwen/Qwen3.5-9B-Base \
        --query_file evaluation/writing_bench/benchmark_query/benchmark_all.jsonl \
        --output_file <dir>/responses/qwen3.5-9b-base/all.jsonl
"""

import argparse
import json
import os
import sys
from concurrent.futures import as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Official WritingBench leaderboard generation params (same as the vLLM driver).
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 16000


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer, query):
    """Mirrors generate_responses_vllm.build_prompt (duplicated rather than
    imported: that module imports vllm at load time)."""
    chat = [{"role": "user", "content": query}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True
        )
    return query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Tinker base_model name")
    parser.add_argument("--hf_tokenizer", type=str, default=None,
                        help="HF tokenizer id if it differs from --model")
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max_in_flight", type=int, default=32,
                        help="concurrent sampling requests kept open against Tinker")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import tinker
    from tinker import types
    from transformers import AutoTokenizer

    queries = load_jsonl(args.query_file)
    if not queries:
        raise SystemExit(f"No queries found in {args.query_file}")

    done = {r["index"] for r in load_jsonl(args.output_file)}
    todo = [q for q in queries if q["index"] not in done]
    print(f"[generate] queries={len(queries)}  done={len(done)}  todo={len(todo)}")
    if not todo:
        print("[generate] Nothing to do — output already complete.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer or args.model)
    stop_ids = sorted({i for i in (tokenizer.eos_token_id,
                                   tokenizer.convert_tokens_to_ids("<|im_end|>"))
                       if isinstance(i, int) and i >= 0})
    client = tinker.ServiceClient().create_sampling_client(base_model=args.model)
    print(f"[generate] Sampling {len(todo)} responses on Tinker ({args.model}) with "
          f"top_p={args.top_p}, top_k={args.top_k}, temp={args.temperature}, "
          f"max_tokens={args.max_tokens}")

    def submit(q):
        ids = tokenizer(build_prompt(tokenizer, q["query"]), add_special_tokens=False).input_ids
        params = types.SamplingParams(
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, stop=stop_ids, seed=args.seed + q["index"])
        return client.sample(prompt=types.ModelInput.from_ints(ids),
                             num_samples=1, sampling_params=params)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    out = open(args.output_file, "a", encoding="utf-8")
    pending, attempts, written, failed = {}, {}, 0, 0
    it = iter(todo)

    def refill():
        while len(pending) < args.max_in_flight:
            q = next(it, None)
            if q is None:
                return
            pending[submit(q)] = q

    refill()
    while pending:
        for fut in as_completed(list(pending)):
            q = pending.pop(fut)
            try:
                res = fut.result()
                text = tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True)
                out.write(json.dumps({"index": q["index"], "response": text}, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
                if written % 25 == 0:
                    print(f"[generate] {written}/{len(todo)} written", flush=True)
            except Exception as e:  # network / server hiccup: retry the row a few times
                n = attempts[q["index"]] = attempts.get(q["index"], 0) + 1
                if n <= 3:
                    print(f"[generate] index {q['index']} attempt {n} failed: {e!r} — retrying", flush=True)
                    pending[submit(q)] = q
                else:
                    failed += 1
                    print(f"[generate] index {q['index']} gave up: {e!r}", flush=True)
            refill()
            break
    out.close()
    print(f"[generate] Wrote {written} records to {args.output_file}"
          + (f" ({failed} failed — rerun to retry)" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
