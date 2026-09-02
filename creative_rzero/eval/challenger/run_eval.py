"""creative_rzero/eval/challenger/run_eval.py — Phase 2: generate against the
frozen eval set (build_dataset.py) with a challenger checkpoint, score the
full metric battery, and write a per-row JSONL + aggregate summary.

Standalone: takes any checkpoint path/HF id and the Hub eval-set repo name,
no orchestrator/config coupling — just files in, files out. W&B logging is
best-effort (mirrors generate_prompts.py::run_generation's guard: only fires
if `wandb` is importable and `WANDB_MODE` isn't "disabled"), not required —
`run()` still returns the summary dict regardless of whether logging fired.

Per-row output schema (one JSON object per line):
    eval_id, domain, subdomain, replicate_idx, guidance_applied,
    format_valid, format_failure_reason,
    query, query_len, criteria, criteria_len,
    domain_adherence, guidance_adherence, criteria_quality,
    judge_backend, judge_reasoning,
    near_duplicate, near_duplicate_partners, near_duplicate_count,
    near_duplicate_similarity, group_self_bleu
Judge/diversity fields are None for format-invalid rows (nothing valid to
judge or compare); `criteria` is also None there. `near_duplicate_partners`
lists the eval_ids of every other row this one is flagged as a near-duplicate
of (not just the closest one), and `near_duplicate_count` is that list's
length — both 0/[] rather than None for format-valid rows with no matches.
`group_self_bleu` is a group-level Self-BLEU score (see diversity.py's
`self_bleu`) repeated across every row in the row's (domain, subdomain)
group — a lexical-diversity read on the whole group, independent of the
`near_duplicate*` threshold-based fields.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from creative_rzero.eval.challenger import challenger_judge_agent  # noqa: E402
from creative_rzero.eval.challenger.aggregate import aggregate_rows  # noqa: E402
from creative_rzero.eval.challenger.diversity import (  # noqa: E402
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDING_SIMILARITY_THRESHOLD,
    near_duplicate_pairs,
    pairwise_bleu_scores,
    semantic_near_duplicate_pairs,
)
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

    #TODO(dayanc): Eval sampling params should be in sync with training config.
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


def _generate_tinker(rows: list[dict], model: str, hf_tokenizer: str = "",
                     seed: int = 42, max_in_flight: int = 32) -> list[str]:
    """Tinker-backend twin of `_load_model`+`_generate`: identical prompt
    construction (chat template, thinking mode) and sampling regime
    (temp 0.6 / top_p 0.95 / top_k 20), but sampling happens on Tinker's
    servers — no local GPU, no vLLM import. One deliberate divergence:
    <|im_end|> joins the stop set alongside eos, since Tinker bills per
    generated token and a base model in a chat template that only stops on
    <|endoftext|> can ramble to the 32k cap. Tokenizer runs locally
    (`hf_tokenizer` defaults to the model id)."""
    import tinker
    from tinker import types as ttypes
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer or model)
    MAX_WINDOW = 32768

    def render(row):
        chat = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ]
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
                add_special_tokens=True, enable_thinking=True,
            )
        return f"system: {row['system_prompt']}\n\nuser: {row['user_prompt']}"

    stop_ids = sorted({i for i in (tokenizer.eos_token_id,
                                   tokenizer.convert_tokens_to_ids("<|im_end|>"))
                       if isinstance(i, int) and i >= 0})
    client = tinker.ServiceClient().create_sampling_client(base_model=model)

    out: dict[int, str] = {}
    attempts: dict[int, int] = {}
    pending: dict = {}
    order = list(range(len(rows)))
    it = iter(order)

    def submit(i):
        ids = tokenizer(render(rows[i]), add_special_tokens=False).input_ids
        params = ttypes.SamplingParams(
            max_tokens=max(256, min(32768, MAX_WINDOW - len(ids) - 16)),
            temperature=0.6, top_p=0.95, top_k=20, stop=stop_ids, seed=seed + i)
        return client.sample(prompt=ttypes.ModelInput.from_ints(ids),
                             num_samples=1, sampling_params=params)

    from concurrent.futures import as_completed

    def refill():
        while len(pending) < max_in_flight:
            i = next(it, None)
            if i is None:
                return
            pending[submit(i)] = i

    refill()
    while pending:
        for fut in as_completed(list(pending)):
            i = pending.pop(fut)
            try:
                res = fut.result()
                out[i] = tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True)
                if len(out) % 25 == 0:
                    print(f"[tinker-gen] {len(out)}/{len(rows)} sampled", flush=True)
            except Exception as e:
                attempts[i] = attempts.get(i, 0) + 1
                if attempts[i] <= 3:
                    print(f"[tinker-gen] row {i} attempt {attempts[i]} failed: {e!r} — retrying", flush=True)
                    pending[submit(i)] = i
                else:
                    raise RuntimeError(f"tinker sampling failed for row {i} after 3 attempts") from e
            refill()
            break
    return [out[i] for i in range(len(rows))]


def _judge_one(agent, row: dict, query: str, criteria: list[dict], judge_retries: int) -> dict:
    """Run a single judge call and normalize both outcomes (parsed result or
    `JudgeParseError`) into the record-field subset `score_rows` merges in —
    lets the parallel map in `score_rows` stay a plain dict update with no
    per-future exception handling of its own."""
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
        return {
            "domain_adherence": judged["domain_adherence"],
            "guidance_adherence": judged["guidance_adherence"],
            "criteria_quality": judged["criteria_quality"],
            "judge_backend": judged["judge_backend"],
            "judge_reasoning": judged["reasoning"],
        }
    except challenger_judge_agent.JudgeParseError as e:
        return {"judge_backend": "error", "judge_reasoning": str(e)}


def score_rows(
    rows: list[dict],
    responses: list[str],
    agent,
    judge_retries: int = 3,
    judge_workers: int = 8,
) -> list[dict]:
    """Format-validate each response (reusing validate_one_shot_response
    unchanged) and, for format-valid rows, run one judge call for
    domain/guidance/criteria-quality. Judge calls are network-bound (a Claude
    API request per row) and independent of each other, so they run
    concurrently across a `ThreadPoolExecutor` of `judge_workers` threads
    rather than one at a time; format validation itself stays sequential
    since it's cheap local parsing. Does not add `near_duplicate` — that's a
    batch-level pass over the whole run, see `add_diversity`."""
    scored = []
    judge_jobs: list[tuple[int, dict, str, list[dict]]] = []
    for row, response in zip(rows, responses):
        is_valid, parsed, _thinking, fmt_reason = validate_one_shot_response(response)
        record = {
            "eval_id": row["eval_id"],
            "domain": row["domain"],
            "subdomain": row["subdomain"],
            "replicate_idx": row["replicate_idx"],
            "guidance_applied": row["guidance_applied"],
            "format_valid": is_valid,
            "format_failure_reason": fmt_reason,
            "query": "",
            "query_len": 0,
            "criteria": None,
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
            record["criteria"] = criteria
            record["criteria_len"] = len(json.dumps(criteria))
            judge_jobs.append((len(scored), row, query, criteria))
        scored.append(record)

    if judge_jobs:
        with ThreadPoolExecutor(max_workers=judge_workers) as pool:
            futures = {
                pool.submit(_judge_one, agent, row, query, criteria, judge_retries): idx
                for idx, row, query, criteria in judge_jobs
            }
            for future in tqdm(as_completed(futures), desc="Scoring rows (judge)", total=len(futures)):
                scored[futures[future]].update(future.result())

    return scored


def add_diversity(
    scored: list[dict], method: str = "embedding"
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Mutates `scored` in place, adding `near_duplicate`,
    `near_duplicate_partners` (eval_ids of *every* other row this one is
    flagged against, not just the closest one), `near_duplicate_count`
    (that list's length), `near_duplicate_similarity` (cosine to the
    strongest match — how strong the match is; borderline near the method's
    threshold, verbatim-level toward 1.0), and `group_self_bleu` (Self-BLEU
    of the row's whole (domain, subdomain) group — see diversity.py's
    `self_bleu` docstring; same value repeated on every row in the group,
    since it's a group-level lexical-diversity stat, not a per-row one) to
    every row. `method` picks the near-duplicate detector: "embedding"
    (Qwen3-Embedding-4B cosine, the default — catches reworded same-task
    duplicates) or "tfidf" (lexical, no model download, the pre-existing
    layer); it does not affect `group_self_bleu`, which is always lexical.
    Grouped by (domain, subdomain) — see diversity.py's module docstring for
    why the comparison is per-group rather than across the whole run.
    Format-invalid rows (no `query` to compare) get None for all five.

    Returns `{(domain, subdomain): {"cosine_similarities": [...], "bleu_scores":
    [...]}}` — every pairwise value computed for that group (not just the
    ones at-or-above the near-duplicate threshold), for callers that want to
    look at a group's full similarity distribution (e.g. a histogram) rather
    than just the flagged pairs and the Self-BLEU mean."""
    pair_fns = {"embedding": semantic_near_duplicate_pairs, "tfidf": near_duplicate_pairs}
    if method not in pair_fns:
        raise ValueError(f"add_diversity method={method!r} must be one of {sorted(pair_fns)}")
    pair_fn = pair_fns[method]
    threshold = {"embedding": EMBEDDING_SIMILARITY_THRESHOLD, "tfidf": DEFAULT_SIMILARITY_THRESHOLD}[method]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in scored:
        r["near_duplicate"] = None
        r["near_duplicate_partners"] = None
        r["near_duplicate_count"] = None
        r["near_duplicate_similarity"] = None
        r["group_self_bleu"] = None
        if r["format_valid"]:
            groups[(r["domain"], r["subdomain"])].append(r)

    group_histograms: dict[tuple[str, str], dict[str, list[float]]] = {}
    for group_key, group_rows in tqdm(list(groups.items()), desc="Diversity groups"):
        queries = [r["query"] for r in group_rows]
        # threshold=-1.0 pulls every pairwise similarity in one pass (below both
        # methods' possible ranges), so near-duplicate flagging and the
        # histogram data below share a single embedding-encode/TF-IDF-fit call
        # instead of computing the same similarities twice.
        all_pairs = pair_fn(queries, threshold=-1.0)
        matches: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for i, j, sim in all_pairs:
            if sim >= threshold:
                matches[i].append((j, sim))
                matches[j].append((i, sim))
        bleu_scores = pairwise_bleu_scores(queries)
        group_bleu = mean(bleu_scores) if bleu_scores else None
        group_histograms[group_key] = {
            "cosine_similarities": [sim for _, _, sim in all_pairs],
            "bleu_scores": bleu_scores,
        }
        for idx, r in enumerate(group_rows):
            row_matches = matches.get(idx, [])
            r["near_duplicate"] = bool(row_matches)
            r["near_duplicate_partners"] = [group_rows[j]["eval_id"] for j, _ in row_matches]
            r["near_duplicate_count"] = len(row_matches)
            r["near_duplicate_similarity"] = (
                round(max(sim for _, sim in row_matches), 3) if row_matches else None
            )
            r["group_self_bleu"] = round(group_bleu, 3) if group_bleu is not None else None
    return group_histograms


def _wandb_active() -> bool:
    return _WANDB_AVAILABLE and _wandb.run is not None


def _flatten_summary(summary: dict) -> dict[str, float]:
    """Flatten aggregate_rows()'s overall stats into a flat metrics dict for
    wandb.log — `challenger_eval/overall/...`. by_domain and by_subdomain are
    deliberately left out here (per-group scalars spawn one panel each and
    clutter the scalar-metrics namespace) — they go into Tables instead, see
    `_group_stats_table`. Skips None values (wandb.log rejects them) and
    non-numeric fields."""
    metrics: dict[str, float] = {}
    for key, value in summary["overall"].items():
        if key == "format_failure_reason_counts":
            for reason, count in value.items():
                metrics[f"challenger_eval/overall/format_failure_reason/{reason}"] = count
        elif isinstance(value, (int, float)):
            metrics[f"challenger_eval/overall/{key}"] = value
    return metrics


def _group_stats_table(groups: dict[str, dict], key_column: str):
    """One row per group (domain or subdomain), one column per stat — a
    single Table panel instead of a scalar panel per group/stat pair."""
    columns = [
        key_column, "n", "format_pass_rate", "domain_adherence_mean",
        "guidance_adherence_mean", "criteria_quality_mean", "duplicate_rate",
        "self_bleu_mean", "query_len_mean", "criteria_len_mean",
    ]
    table = _wandb.Table(columns=columns)
    for group_key, stats in groups.items():
        table.add_data(*(stats.get(c) if c != key_column else group_key for c in columns))
    return table


def _rows_table(scored: list[dict]):
    columns = [
        "eval_id", "domain", "subdomain", "replicate_idx", "guidance_applied", "format_valid",
        "format_failure_reason", "domain_adherence", "guidance_adherence",
        "criteria_quality", "judge_backend", "near_duplicate", "near_duplicate_partners",
        "near_duplicate_count", "near_duplicate_similarity", "group_self_bleu", "query", "criteria",
    ]
    table = _wandb.Table(columns=columns)
    for r in scored:
        table.add_data(
            r["eval_id"], r["domain"], r["subdomain"], r["replicate_idx"],
            "; ".join(r["guidance_applied"]) if r["guidance_applied"] else "",
            r["format_valid"], r["format_failure_reason"], r["domain_adherence"],
            r["guidance_adherence"], r["criteria_quality"], r["judge_backend"],
            r["near_duplicate"],
            "; ".join(str(p) for p in r["near_duplicate_partners"]) if r["near_duplicate_partners"] else "",
            r["near_duplicate_count"], r["near_duplicate_similarity"], r["group_self_bleu"],
            r["query"], json.dumps(r["criteria"]) if r["criteria"] is not None else "",
        )
    return table


def _histogram_image(values: list[float], title: str):
    """Render `values` as a small matplotlib histogram and wrap it as a
    `wandb.Image` — lazily imports matplotlib (Agg backend, headless) since
    it's only needed on this W&B-logging path, mirroring the
    vllm/transformers/sentence-transformers lazy-import pattern used
    elsewhere in this module. Renders an empty axes (no error) for an empty
    `values` list, e.g. a group with fewer than 2 valid rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.hist(values, bins=15)
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    image = _wandb.Image(fig)
    plt.close(fig)
    return image


def _similarity_histogram_table(group_histograms: dict[tuple[str, str], dict[str, list[float]]]):
    """One row per (domain, subdomain) group: pairwise cosine-similarity and
    pairwise Self-BLEU histograms rendered as images, plus each distribution's
    mean as a sortable scalar summary of the group."""
    columns = [
        "domain", "subdomain", "mean_cosine_score", "max_cosine_score", "cosine_similarity_hist",
        "mean_bleu_score", "max_bleu_score", "self_bleu_hist",
    ]
    table = _wandb.Table(columns=columns)
    for (domain, subdomain), data in sorted(group_histograms.items()):
        cosine_sims = data["cosine_similarities"]
        bleu_scores = data["bleu_scores"]
        table.add_data(
            domain, subdomain,
            round(mean(cosine_sims), 3) if cosine_sims else None,
            round(max(cosine_sims), 3) if cosine_sims else None,
            _histogram_image(cosine_sims, f"{domain}::{subdomain} cosine sim"),
            round(mean(bleu_scores), 3) if bleu_scores else None,
            round(max(bleu_scores), 3) if bleu_scores else None,
            _histogram_image(bleu_scores, f"{domain}::{subdomain} self-BLEU"),
        )
    return table


def _log_to_wandb(
    *,
    model: str,
    summary: dict,
    scored: list[dict],
    group_histograms: dict[tuple[str, str], dict[str, list[float]]],
    step: int | None,
    wandb_group: str,
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
    metrics["challenger_eval/by_domain"] = _group_stats_table(summary["by_domain"], "domain")
    metrics["challenger_eval/by_subdomain"] = _group_stats_table(summary["by_subdomain"], "subdomain")
    metrics["challenger_eval/rows"] = _rows_table(scored)
    metrics["challenger_eval/similarity_histograms"] = _similarity_histogram_table(group_histograms)
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
    dup_method: str = "embedding",
    judge_workers: int = 8,
    gen_backend: str = "vllm",
    hf_tokenizer: str = "",
) -> dict:
    rows = load_eval_set(dataset_repo, local_path, limit=limit)

    if gen_backend == "tinker":
        responses = _generate_tinker(rows, model, hf_tokenizer=hf_tokenizer, seed=seed)
    elif gen_backend == "vllm":
        llm, tokenizer = _load_model(model, seed=seed)
        responses = _generate(llm, tokenizer, rows)
        # Free vLLM's GPU reservation before scoring: the embedding dup detector
        # loads Qwen3-Embedding-4B (~8GB) on the same device.
        del llm
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    else:
        raise ValueError(f"gen_backend={gen_backend!r} must be 'vllm' or 'tinker'")


    print("Initializing Claude Judge Agent")
    agent = challenger_judge_agent.get_agent(judge_type)
    print("Initialized Claude Judge Agent")
    print("Scoring rows")
    scored = score_rows(rows, responses, agent, judge_workers=judge_workers)
    print("Done Scoring Rows")
    print("Running Diversity Eval")
    group_histograms = add_diversity(scored, method=dup_method)
    print("Completed Diversity Eval")
    print(f"Writing results to output path: {out_path}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in scored) + "\n")
    print("Creating Summary")
    summary = aggregate_rows(scored)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print("Done Creating Summary")
    print(f"Wrote {len(scored)} scored rows to {out_path}")
    print(f"Wrote aggregate summary to {summary_path}")

    if _WANDB_AVAILABLE and os.getenv("WANDB_MODE") != "disabled":
        _log_to_wandb(
            model=model, summary=summary, scored=scored, group_histograms=group_histograms,
            step=step, wandb_group=wandb_group,
        )

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
    parser.add_argument(
        "--dup-method", type=str, default="embedding", choices=("embedding", "tfidf"),
        help="near-duplicate detector: Qwen3-Embedding-4B cosine (default) or lexical TF-IDF",
    )
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step, used as the wandb.log x-axis")
    parser.add_argument("--wandb-group", type=str, default="", help="wandb run group (e.g. the coevolve run abbr)")
    parser.add_argument(
        "--gen-backend", type=str, default="vllm", choices=("vllm", "tinker"),
        help="where generation runs: local vLLM (GPU) or the Tinker sampling API",
    )
    parser.add_argument("--hf-tokenizer", type=str, default="",
                        help="tinker backend: HF tokenizer id when it differs from --model")
    parser.add_argument(
        "--judge-workers", type=int, default=8,
        help="concurrent judge API calls (score_generated_prompt is network-bound and independent per row)",
    )
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
        dup_method=args.dup_method,
        judge_workers=args.judge_workers,
        gen_backend=args.gen_backend,
        hf_tokenizer=args.hf_tokenizer,
    )
