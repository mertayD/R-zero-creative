"""
Deterministic stratified subset sampler for WritingBench.

Produces two reproducible subsets:
    smoke50.jsonl  — 50 prompts; for end-to-end smoke runs and CI-style checks.
    mid100.jsonl   — 100 prompts; for between-iteration evals during R-Zero
                     training, where the full 1,000-prompt benchmark would be
                     too expensive.

Both are stratified by `domain1` with allocations roughly proportional to the
full benchmark (Finance > Politics > Literature > Academic > Advertising > Edu).
Sampling within a stratum is shuffled by a fixed seed (42) so the subsets are
stable across machines.

The smoke50/mid100 subsets are committed to the repo so eval is reproducible
without re-running this script. Re-run only if you want to change the
allocation or add a new subset.

Usage:
    python evaluation/writing_bench/make_subsets.py \
        --benchmark_all evaluation/writing_bench/benchmark_query/benchmark_all.jsonl \
        --out_dir evaluation/writing_bench/subsets
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict


# Rounded so the totals hit exactly 50 / 100.
SMOKE_50 = {
    'Finance & Business':     11,
    'Politics & Law':         10,
    'Literature & Arts':      9,
    'Academic & Engineering': 8,
    'Advertising & Marketing': 6,
    'Education':              6,
}

MID_100 = {
    'Finance & Business':     21,
    'Politics & Law':         20,
    'Literature & Arts':      18,
    'Academic & Engineering': 17,
    'Advertising & Marketing': 13,
    'Education':              11,
}

SUBSETS = {
    'smoke50': (SMOKE_50, 42),
    'mid100':  (MID_100,  42),
}


def sample(rows, allocation, seed):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r['domain1']].append(r)

    rng = random.Random(seed)
    picked = []
    for domain, n in allocation.items():
        pool = sorted(buckets[domain], key=lambda r: r['index'])  # stable order
        rng.shuffle(pool)
        if n > len(pool):
            raise ValueError(
                f"Allocation requests {n} from {domain} but only {len(pool)} available")
        picked.extend(pool[:n])
    picked.sort(key=lambda r: r['index'])
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_all", required=True,
                   help="Path to upstream benchmark_query/benchmark_all.jsonl")
    p.add_argument("--out_dir", required=True,
                   help="Directory where smoke50.jsonl and mid100.jsonl are written")
    args = p.parse_args()

    with open(args.benchmark_all, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"[make_subsets] Loaded {len(rows)} rows from {args.benchmark_all}")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, (allocation, seed) in SUBSETS.items():
        chosen = sample(rows, allocation, seed)
        out_path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            for r in chosen:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        d1 = Counter(r['domain1'] for r in chosen)
        lang = Counter(r['lang'] for r in chosen)
        print(f"[make_subsets] {name}: n={len(chosen)}  lang={dict(lang)}  d1={dict(d1)}  -> {out_path}")


if __name__ == "__main__":
    main()
