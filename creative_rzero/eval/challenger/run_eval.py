"""creative_rzero/eval/challenger/run_eval.py — Phase 2: generate against the
frozen eval set (build_dataset.py) with a challenger checkpoint, score the
full metric battery, and write a per-row JSONL + aggregate summary.

Standalone: takes any checkpoint path/HF id and the Hub eval-set repo name,
no orchestrator/config coupling — just files in, files out. W&B logging is
best-effort (mirrors generate_prompts.py::run_generation's guard: only fires
if `wandb` is importable and `WANDB_MODE` isn't "disabled"), not required —
`run()` still returns the summary dict regardless of whether logging fired.

Per-row output schema (one JSON object per line):
    eval_id, domain, subdomain, replicate_idx,
    format_valid, format_failure_reason,
    query, query_len, criteria_len,
    domain_adherence, guidance_adherence, criteria_quality,
    judge_backend, judge_reasoning,
    near_duplicate
Judge/diversity fields are None for format-invalid rows (nothing valid to
judge or compare).

`vllm`/`transformers` are imported lazily inside `_load_model`/`_generate`,
mirroring generate_prompts.py::run_generation, so importing this module (for
its scoring/aggregation functions, e.g. from a test) never requires a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from creative_rzero.eval.challenger import challenger_judge_agent  # noqa: E402
from creative_rzero.eval.challenger.aggregate import aggregate_rows  # noqa: E402
from creative_rzero.eval.challenger.diversity import near_duplicate_pairs  # noqa: E402
from creative_rzero.steps.generate_prompts import validate_one_shot_response  # noqa: E402

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def load_eval_set(
    dataset_repo: str | None = None,
    local_path: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Load the frozen eval set — from the Hub (`dataset_repo`, the normal
    path) or a local JSONL (`local_path`, for fast iteration against
    build_dataset.py's --dry-run output). Exactly one must be given."""
    if dataset_repo:
        from datasets import load_dataset

        rows = list(load_dataset(dataset_repo, split="eval"))
    elif local_path:
        text = Path(local_path).read_text()
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raise ValueError("one of dataset_repo/local_path is required")

    return rows[:limit] if limit is not None else rows


def _load_model(model: str, seed: int = 42):
    import vllm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = vllm.LLM(model=model, tokenizer=model, seed=seed)
    return llm, tokenizer


def _generate(llm, tokenizer, rows: list[dict]) -> list[str]:
    """One completion per row, using each row's already-rendered
    system/user prompt from build_dataset.py — no DomainSampler or
    prompt-building here, the eval set is frozen. Sampling params mirror
    generate_prompts.py::generate_prompts_batch's Qwen3 thinking-mode
    settings (no greedy decoding)."""
    import vllm

    prompts = []
    for row in rows:
        chat = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ]
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True,
                enable_thinking=True,
            )
        else:
            prompt = f"system: {row['system_prompt']}\n\nuser: {row['user_prompt']}"
        prompts.append(prompt)

    sampling_params = vllm.SamplingParams(
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    completions = llm.generate(prompts, sampling_params=sampling_params)
    return [c.outputs[0].text for c in completions]


def score_rows(rows: list[dict], responses: list[str], agent, judge_retries: int = 3) -> list[dict]:
    """Format-validate each response (reusing validate_one_shot_response
    unchanged) and, for format-valid rows, run one judge call for
    domain/guidance/criteria-quality. Does not add `near_duplicate` — that's
    a batch-level pass over the whole run, see `add_diversity`."""
    scored = []
    for row, response in zip(rows, responses):
        is_valid, parsed, _thinking, fmt_reason = validate_one_shot_response(response)
        record = {
            "eval_id": row["eval_id"],
            "domain": row["domain"],
            "subdomain": row["subdomain"],
            "replicate_idx": row["replicate_idx"],
            "format_valid": is_valid,
            "format_failure_reason": fmt_reason,
            "query": "",
            "query_len": 0,
            "criteria_len": 0,
            "domain_adherence": None,
            "guidance_adherence": None,
            "criteria_quality": None,
            "judge_backend": None,
            "judge_reasoning": None,
        }
        if is_valid:
            query = parsed.get("query", "")
            criteria = parsed.get("criteria", [])
            record["query"] = query
            record["query_len"] = len(query)
            record["criteria_len"] = len(json.dumps(criteria))
            try:
                judged = challenger_judge_agent.score_generated_prompt(
                    agent,
                    domain_name=row["domain_name"],
                    subdomain=row["subdomain"],
                    guidance_applied=row["guidance_applied"],
                    query=query,
                    criteria=criteria,
                    max_retries=judge_retries,
                )
                record["domain_adherence"] = judged["domain_adherence"]
                record["guidance_adherence"] = judged["guidance_adherence"]
                record["criteria_quality"] = judged["criteria_quality"]
                record["judge_backend"] = judged["judge_backend"]
                record["judge_reasoning"] = judged["reasoning"]
            except challenger_judge_agent.JudgeParseError as e:
                record["judge_backend"] = "error"
                record["judge_reasoning"] = str(e)
        scored.append(record)
    return scored


def add_diversity(scored: list[dict]) -> None:
    """Mutates `scored` in place, adding `near_duplicate` to every row.
    Grouped by (domain, subdomain) — see diversity.py's module docstring for
    why the comparison is per-group rather than across the whole run.
    Format-invalid rows (no `query` to compare) get `near_duplicate = None`."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in scored:
        r["near_duplicate"] = None
        if r["format_valid"]:
            groups[(r["domain"], r["subdomain"])].append(r)

    for group_rows in groups.values():
        queries = [r["query"] for r in group_rows]
        flagged: set[int] = set()
        for i, j, _sim in near_duplicate_pairs(queries):
            flagged.add(i)
            flagged.add(j)
        for idx, r in enumerate(group_rows):
            r["near_duplicate"] = idx in flagged


def _wandb_active() -> bool:
    return _WANDB_AVAILABLE and _wandb.run is not None


def _flatten_summary(summary: dict) -> dict[str, float]:
    """Flatten aggregate_rows()'s {overall, by_domain, by_subdomain} summary
    into a flat metrics dict for wandb.log — `challenger_eval/overall/...`,
    `challenger_eval/by_domain/<D>/...`. by_subdomain is deliberately left
    out here (76 subdomains x ~8 stats would clutter the scalar-metrics
    namespace) — it goes into a Table instead, see `_by_subdomain_table`.
    Skips None values (wandb.log rejects them) and non-numeric fields."""
    metrics: dict[str, float] = {}

    def _add(prefix: str, stats: dict) -> None:
        for key, value in stats.items():
            if key == "format_failure_reason_counts":
                for reason, count in value.items():
                    metrics[f"{prefix}/format_failure_reason/{reason}"] = count
            elif isinstance(value, (int, float)):
                metrics[f"{prefix}/{key}"] = value

    _add("challenger_eval/overall", summary["overall"])
    for domain, stats in summary["by_domain"].items():
        _add(f"challenger_eval/by_domain/{domain}", stats)
    return metrics


def _by_subdomain_table(summary: dict):
    columns = [
        "subdomain", "n", "format_pass_rate", "domain_adherence_mean",
        "guidance_adherence_mean", "criteria_quality_mean", "duplicate_rate",
        "query_len_mean", "criteria_len_mean",
    ]
    table = _wandb.Table(columns=columns)
    for subdomain, stats in summary["by_subdomain"].items():
        table.add_data(*(stats.get(c) if c != "subdomain" else subdomain for c in columns))
    return table


def _rows_table(scored: list[dict]):
    columns = [
        "eval_id", "domain", "subdomain", "replicate_idx", "format_valid",
        "format_failure_reason", "domain_adherence", "guidance_adherence",
        "criteria_quality", "judge_backend", "near_duplicate", "query_preview",
    ]
    table = _wandb.Table(columns=columns)
    for r in scored:
        table.add_data(
            r["eval_id"], r["domain"], r["subdomain"], r["replicate_idx"], r["format_valid"],
            r["format_failure_reason"], r["domain_adherence"], r["guidance_adherence"],
            r["criteria_quality"], r["judge_backend"], r["near_duplicate"], r["query"][:200],
        )
    return table


def _log_to_wandb(
    *, model: str, summary: dict, scored: list[dict], step: int | None, wandb_group: str
) -> None:
    """Log this run's summary + per-row/per-subdomain tables to W&B.
    Attaches to an already-active run (e.g. the orchestrator's, if this is
    invoked from within one) when `WANDB_RUN_ID` is set — otherwise creates
    and owns a standalone run, mirroring generate_prompts.py::run_generation's
    exact owned-vs-attached pattern."""
    wandb_parent_run_id = os.getenv("WANDB_RUN_ID")
    owned = False
    if wandb_parent_run_id:
        _wandb.init(project=os.getenv("WANDB_PROJECT", "r-zero-creative"), id=wandb_parent_run_id, resume="allow")
    else:
        _wandb.init(
            project=os.getenv("WANDB_PROJECT", "r-zero-creative"),
            job_type="challenger-eval",
            group=wandb_group or None,
            name=f"challenger-eval_{Path(model).name or model}",
            config={"model": model, "n_rows": len(scored)},
        )
        owned = True

    metrics = _flatten_summary(summary)
    metrics["challenger_eval/by_subdomain"] = _by_subdomain_table(summary)
    metrics["challenger_eval/rows"] = _rows_table(scored)
    _wandb.log(metrics, step=step)

    if owned:
        _wandb.finish()


def run(
    *,
    model: str,
    out_path: str | Path,
    dataset_repo: str | None = None,
    local_path: str | None = None,
    judge_type: str = "claude",
    limit: int | None = None,
    seed: int = 42,
    step: int | None = None,
    wandb_group: str = "",
) -> dict:
    rows = load_eval_set(dataset_repo, local_path, limit=limit)

    llm, tokenizer = _load_model(model, seed=seed)
    responses = _generate(llm, tokenizer, rows)

    agent = challenger_judge_agent.get_agent(judge_type)
    scored = score_rows(rows, responses, agent)
    add_diversity(scored)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in scored) + "\n")

    summary = aggregate_rows(scored)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {len(scored)} scored rows to {out_path}")
    print(f"Wrote aggregate summary to {summary_path}")

    if _WANDB_AVAILABLE and os.getenv("WANDB_MODE") != "disabled":
        _log_to_wandb(model=model, summary=summary, scored=scored, step=step, wandb_group=wandb_group)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the challenger eval harness against a checkpoint")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint path or HF model id")
    parser.add_argument("--dataset-repo", type=str, default=None, help="HF Hub repo id for the frozen eval set")
    parser.add_argument("--local-path", type=str, default=None, help="Local JSONL eval-set path (dev/testing)")
    parser.add_argument(
        "--judge-type", type=str, default="claude", choices=list(challenger_judge_agent.VALID_JUDGE_TYPES)
    )
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N rows (smoke runs)")
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step, used as the wandb.log x-axis")
    parser.add_argument("--wandb-group", type=str, default="", help="wandb run group (e.g. the coevolve run abbr)")
    args = parser.parse_args()

    if not args.dataset_repo and not args.local_path:
        parser.error("one of --dataset-repo/--local-path is required")

    run(
        model=args.model,
        out_path=args.out_path,
        dataset_repo=args.dataset_repo,
        local_path=args.local_path,
        judge_type=args.judge_type,
        limit=args.limit,
        seed=args.seed,
        step=args.step,
        wandb_group=args.wandb_group,
    )
