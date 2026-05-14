"""
Postprocess helper for WritingBench eval runs.

Joins three files keyed by `index`:
    - the subset jsonl  (prompt + per-criterion checklist + domain1 + domain2 + lang)
    - responses.jsonl   (model output)
    - scores/<model_slug>.jsonl  (judge score and reason per criterion)

See SKILL.md in this directory for usage. CLI entrypoints:

    inspect.py one    --run <run_dir> --index <i>
    inspect.py list   --run <run_dir> [--sort score|index] [--domain X] [--lang L]
    inspect.py worst  --run <run_dir> [--n 10]
    inspect.py compare --run-a <dir> --run-b <dir>

Programmatic API:
    load_run(run_dir, subset=None) -> dict[int, dict]
    join_run(records)              -> list[dict]   (flat, DataFrame-friendly)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WB_DIR = os.path.join(_REPO_ROOT, "evaluation", "writing_bench")


def _subset_jsonl_path(subset: str) -> str:
    """Resolve the subset's source jsonl (prompts + checklists)."""
    if subset == "full":
        return os.path.join(WB_DIR, "benchmark_query", "benchmark_all.jsonl")
    return os.path.join(WB_DIR, "subsets", f"{subset}.jsonl")


def _read_jsonl(path: str) -> List[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_score_file(run_dir: str) -> str:
    candidates = sorted(glob(os.path.join(run_dir, "scores", "*.jsonl")))
    if not candidates:
        raise FileNotFoundError(
            f"No score file found under {run_dir}/scores/. "
            "Has stage 2 (judging) finished?"
        )
    if len(candidates) > 1:
        # Multiple model slugs in the same run dir is unusual but allowed —
        # take the one whose stem matches the parent dir name (the model slug).
        slug = os.path.basename(os.path.normpath(run_dir).rstrip(os.sep))
        for c in candidates:
            if os.path.splitext(os.path.basename(c))[0] == slug:
                return c
    return candidates[0]


# ---------------------------------------------------------------------------
# Loading and joining
# ---------------------------------------------------------------------------

def load_run(run_dir: str, subset: Optional[str] = None) -> Dict[int, dict]:
    """
    Load and join all three files for one run.

    Returns a dict keyed by index. Each value carries:
        prompt, domain1, domain2, lang, checklist, response, scores, avg
    where `scores` is the raw {criterion_name: [{score, reason}, ...]} mapping
    and `avg` is the mean of all criterion means (matches the upstream
    calculate_scores.py per-prompt average).
    """
    run_dir = os.path.abspath(run_dir)
    subset = subset or os.path.basename(os.path.normpath(run_dir))
    subset_path = _subset_jsonl_path(subset)

    subset_rows  = _read_jsonl(subset_path)
    response_rows = _read_jsonl(os.path.join(run_dir, "responses.jsonl"))
    score_rows   = _read_jsonl(_find_score_file(run_dir))

    by_idx: Dict[int, dict] = {}
    for r in subset_rows:
        by_idx[r["index"]] = {
            "index":     r["index"],
            "prompt":    r["query"],
            "domain1":   r.get("domain1"),
            "domain2":   r.get("domain2"),
            "lang":      r.get("lang"),
            "checklist": r.get("checklist", []),
            "response":  None,
            "scores":    None,
            "avg":       None,
        }

    for r in response_rows:
        if r["index"] in by_idx:
            by_idx[r["index"]]["response"] = r.get("response")

    for r in score_rows:
        idx = r["index"]
        if idx not in by_idx:
            continue
        scores = r.get("scores", {})
        by_idx[idx]["scores"] = scores
        per_criterion_means = []
        for _, evals in scores.items():
            if not evals:
                continue
            per_criterion_means.append(
                sum(e["score"] for e in evals) / len(evals)
            )
        if per_criterion_means:
            by_idx[idx]["avg"] = sum(per_criterion_means) / len(per_criterion_means)

    return by_idx


def join_run(records: Dict[int, dict]) -> List[dict]:
    """Flatten the dict-of-records into a list-of-dicts (DataFrame-ready)."""
    rows = []
    for idx in sorted(records.keys()):
        rec = records[idx]
        rows.append({
            "index":   idx,
            "domain1": rec["domain1"],
            "domain2": rec["domain2"],
            "lang":    rec["lang"],
            "avg":     rec["avg"],
            "prompt":  rec["prompt"],
            "response": rec["response"],
        })
    return rows


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _truncate(text: Optional[str], limit: int) -> str:
    if text is None:
        return "<missing>"
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated to {limit} chars; pass --full to see all]"


def _print_one(rec: dict, char_limit: int) -> None:
    idx = rec["index"]
    print(f"=== index {idx}  |  {rec['domain1']} > {rec['domain2']}  |  lang={rec['lang']}  |  avg={rec['avg']} ===")
    print()
    print("--- Prompt ---")
    print(_truncate(rec["prompt"], char_limit))
    print()
    print("--- Response ---")
    print(_truncate(rec["response"], char_limit))
    print()
    print("--- Scores ---")
    scores = rec.get("scores") or {}
    for criterion, evals in scores.items():
        if not evals:
            continue
        e = evals[0]  # Upstream WritingBench evaluates each criterion once by default.
        score = e.get("score")
        reason = e.get("reason", "")
        print(f"  [{score:>2}] {criterion}")
        # Indent the reason for readability.
        for line in _truncate(reason, char_limit).splitlines():
            print(f"        {line}")
    print()


def _print_list(records: Iterable[dict], sort: str = "score") -> None:
    rows = [r for r in records if r.get("avg") is not None]
    if sort == "score":
        rows.sort(key=lambda r: r["avg"])
    else:
        rows.sort(key=lambda r: r["index"])

    print(f"{'idx':>4}  {'lang':>4}  {'avg':>5}  {'domain1':<24}  {'domain2':<24}  scores")
    print("-" * 100)
    for r in rows:
        scores_repr = "?"
        if r.get("scores"):
            per_crit = []
            for evals in r["scores"].values():
                if evals:
                    per_crit.append(str(evals[0].get("score", "?")))
            scores_repr = ",".join(per_crit)
        d1 = (r["domain1"] or "")[:24]
        d2 = (r["domain2"] or "")[:24]
        print(f"{r['index']:>4}  {r['lang'] or '':>4}  {r['avg']:5.2f}  {d1:<24}  {d2:<24}  {scores_repr}")


def _print_compare(records_a: Dict[int, dict], records_b: Dict[int, dict],
                   label_a: str, label_b: str) -> None:
    common = sorted(set(records_a) & set(records_b))
    rows = []
    for idx in common:
        a = records_a[idx]["avg"]
        b = records_b[idx]["avg"]
        if a is None or b is None:
            continue
        rows.append({"idx": idx, "a": a, "b": b, "delta": b - a,
                     "domain1": records_a[idx]["domain1"]})
    rows.sort(key=lambda r: r["delta"])

    if not rows:
        print("No overlapping indices with non-null scores between the two runs.")
        return

    print(f"{'idx':>4}  {label_a:>10}  {label_b:>10}  {'delta':>8}  domain1")
    print("-" * 64)
    for r in rows:
        sign = "+" if r["delta"] >= 0 else ""
        print(f"{r['idx']:>4}  {r['a']:10.2f}  {r['b']:10.2f}  {sign}{r['delta']:7.2f}  {r['domain1']}")
    print("-" * 64)
    avg_a = sum(r["a"] for r in rows) / len(rows)
    avg_b = sum(r["b"] for r in rows) / len(rows)
    sign = "+" if (avg_b - avg_a) >= 0 else ""
    print(f"{'mean':>4}  {avg_a:10.2f}  {avg_b:10.2f}  {sign}{(avg_b - avg_a):7.2f}  ({len(rows)} prompts)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _filter(records: Dict[int, dict], domain: Optional[str], lang: Optional[str]) -> List[dict]:
    out = []
    for r in records.values():
        if domain and r.get("domain1") != domain:
            continue
        if lang and r.get("lang") != lang:
            continue
        out.append(r)
    return out


def _add_run_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True,
                   help="Run directory: <storage_path>/writing_bench/<model_slug>/<subset>/")
    p.add_argument("--subset", default=None,
                   help="Override subset name. Default: inferred from --run dir name.")
    p.add_argument("--full", action="store_true",
                   help="Print full prompt/response/reason text without truncation.")


def main() -> int:
    parser = argparse.ArgumentParser(prog="inspect.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_one = sub.add_parser("one", help="Print one prompt's full prompt/response/scores.")
    _add_run_arg(p_one)
    p_one.add_argument("--index", type=int, required=True)

    p_list = sub.add_parser("list", help="Table of every prompt with its average score.")
    _add_run_arg(p_list)
    p_list.add_argument("--sort", choices=["score", "index"], default="score")
    p_list.add_argument("--domain", default=None,
                        help='Filter by domain1, e.g. "Education".')
    p_list.add_argument("--lang", default=None, choices=["en", "zh"])

    p_worst = sub.add_parser("worst", help="Show the N lowest-scoring prompts.")
    _add_run_arg(p_worst)
    p_worst.add_argument("--n", type=int, default=10)

    p_cmp = sub.add_parser("compare", help="Diff two runs on the same subset.")
    p_cmp.add_argument("--run-a", required=True)
    p_cmp.add_argument("--run-b", required=True)
    p_cmp.add_argument("--subset", default=None,
                       help="Override subset (default: inferred from --run-a).")

    args = parser.parse_args()
    char_limit = 0 if getattr(args, "full", False) else 2000

    if args.cmd == "one":
        records = load_run(args.run, args.subset)
        if args.index not in records:
            print(f"Index {args.index} not found in {args.run}.", file=sys.stderr)
            return 2
        _print_one(records[args.index], char_limit)
        return 0

    if args.cmd == "list":
        records = load_run(args.run, args.subset)
        rows = _filter(records, args.domain, args.lang)
        if not rows:
            print("No rows match the filter.")
            return 0
        _print_list(rows, sort=args.sort)
        return 0

    if args.cmd == "worst":
        records = load_run(args.run, args.subset)
        rows = [r for r in records.values() if r.get("avg") is not None]
        rows.sort(key=lambda r: r["avg"])
        for r in rows[: args.n]:
            _print_one(r, char_limit)
        return 0

    if args.cmd == "compare":
        a = load_run(args.run_a, args.subset)
        b = load_run(args.run_b, args.subset)
        label_a = os.path.basename(os.path.dirname(os.path.abspath(args.run_a)))
        label_b = os.path.basename(os.path.dirname(os.path.abspath(args.run_b)))
        _print_compare(a, b, label_a or "A", label_b or "B")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
