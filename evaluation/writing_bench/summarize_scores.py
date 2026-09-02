"""
Summarize a WritingBench scores file (evaluate_benchmark.py output) into the
headline numbers the leaderboard and our earlier baselines report: overall mean
criterion score (1-10) and its x10 percentage, per-domain means, query count,
and how many (response, criterion) pairs never got a valid judgment.

Aggregation is the mean over ALL criterion scores (not the mean of per-query
means) — identical to calculate_scores.py's overall and to the Modal-era
baselines, so numbers stay comparable across generation backends.

    python evaluation/writing_bench/summarize_scores.py \
        --scores_file <dir>/scores/qwen3.5-9b-base/all.jsonl \
        --benchmark_file evaluation/writing_bench/benchmark_query/benchmark_all.jsonl \
        [--wandb_name wb-qwen3.5-9b-base-tinker]
"""

import argparse
import json
import os


def summarize(scores_file, benchmark_file):
    bench = {r["index"]: r for r in (json.loads(l) for l in open(benchmark_file, encoding="utf-8") if l.strip())}
    per_query, dom_sum, dom_n, all_sum, all_n, missing = {}, {}, {}, 0.0, 0, 0
    for line in open(scores_file, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        vals = [e["score"] for evs in rec["scores"].values() for e in evs]
        missing += sum(1 for evs in rec["scores"].values() if not evs)
        if not vals:
            continue
        per_query[rec["index"]] = sum(vals) / len(vals)
        all_sum += sum(vals)
        all_n += len(vals)
        d = bench[rec["index"]]["domain1"]
        dom_sum[d] = dom_sum.get(d, 0.0) + sum(vals)
        dom_n[d] = dom_n.get(d, 0) + len(vals)
    return {
        "n_queries": len(per_query),
        "overall": round(all_sum / max(all_n, 1), 3),
        "overall_pct": round(10 * all_sum / max(all_n, 1), 2),
        "criteria_missing": missing,
        **{f"domain/{d}": round(dom_sum[d] / dom_n[d], 3) for d in sorted(dom_n)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_file", required=True)
    ap.add_argument("--benchmark_file", required=True)
    ap.add_argument("--wandb_name", default=None, help="log the summary as a W&B run with this name")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "r-zero-creative"))
    args = ap.parse_args()

    summary = summarize(args.scores_file, args.benchmark_file)
    print(json.dumps(summary, indent=2))
    if args.wandb_name:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, job_type="writingbench")
        run.summary.update(summary)
        run.finish()
        print(f"[summarize] logged to W&B run {args.wandb_name}")


if __name__ == "__main__":
    main()
