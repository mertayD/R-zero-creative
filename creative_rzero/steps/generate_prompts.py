"""creative_rzero/steps/generate_prompts.py — challenger checkpoint -> WritingPrompt JSON.

Owns the whole one-shot prompt-generation flow: domain/subdomain sampling,
prompt construction, response validation, the batch-generation retry loop,
the CLI entrypoint, and the typed pipeline step (`generate_prompts`) the
orchestrator calls. Consolidates what used to be split across
`question_generate/one_shot_creative_question_generate.py` (all of the
above) and this file (a subprocess shim around it) — see
docs/REFACTOR_PLAN.md T2.4.

`vllm`/`transformers` are imported lazily, inside `run_generation`, rather
than at module scope: this file's data model (`creative_rzero.data`) and
validation utilities (`creative_rzero.utils`) are consumed by
`steps/build_parquet.py` and the reward callers, none of which need a GPU,
and `creative_rzero/config.py`'s docstring explains why that matters (fast,
GPU-free config validation/tests).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from creative_rzero.config import ExperimentConfig
from creative_rzero.data.writing_prompt import PromptBatch, QueryRequirements, WritingPrompt
from creative_rzero.paths import RunPaths
from creative_rzero.prompts.one_shot import ONE_SHOT_SYSTEM_PROMPT, render_one_shot_user_prompt
from creative_rzero.failure_reasons import FormatFailureReason
from creative_rzero.utils import FormatValidator
from evaluation.shared.utilities import split_thinking
from question_generate.creative_writing_prompts import QUERY_REFINEMENT_GUIDANCE_POOL, WRITING_DOMAINS

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PromptGenerationError(RuntimeError):
    """Raised when generation fails to load its model or under-delivers prompts."""


def _wandb_active() -> bool:
    return _WANDB_AVAILABLE and _wandb.run is not None


# =============================================================================
# Domain & Subdomain Sampling
# =============================================================================

class DomainSampler:
    """Handles uniform sampling of domains and subdomains from WritingBench."""

    def __init__(self, seed: Optional[int] = 42):
        if seed is not None:
            random.seed(seed)
        self.domains = list(WRITING_DOMAINS.keys())  # D1, D2, D3, D4, D5, D6

    def sample_domain_subdomain_pair(self) -> tuple[str, str]:
        """
        Uniformly sample a domain and subdomain pair.

        Returns:
            Tuple of (domain_key, subdomain_name)
        """
        domain_key = random.choice(self.domains)
        subdomains = WRITING_DOMAINS[domain_key]['subdomains']
        subdomain = random.choice(subdomains)
        return domain_key, subdomain


# =============================================================================
# One-Shot Prompt Generation
# =============================================================================

def build_one_shot_prompt(domain_key: str, subdomain: str, language: str = "English") -> tuple[str, str, list]:
    """
    Build the one-shot prompt with all refinement logic and detailed criteria structure.

    Args:
        domain_key: Domain identifier (D1-D6)
        subdomain: Subdomain name
        language: Language for generation

    Returns:
        Tuple of (system_prompt, user_prompt, applied_guidance)
    """
    domain_name = WRITING_DOMAINS[domain_key]['name']
    domain_description = WRITING_DOMAINS[domain_key]['description']

    # Select guidance items to apply (1-6 items like multi-step)
    num_guidance = random.randint(1, len(QUERY_REFINEMENT_GUIDANCE_POOL))
    num_guidance = min(num_guidance, len(QUERY_REFINEMENT_GUIDANCE_POOL))
    applied_guidance = random.sample(QUERY_REFINEMENT_GUIDANCE_POOL, num_guidance) if num_guidance > 0 else []
    guidance_text = "\n".join([f"  - {g}" for g in applied_guidance]) if applied_guidance else "  (none selected)"

    user_prompt = render_one_shot_user_prompt(
        domain_name=domain_name,
        domain_description=domain_description,
        subdomain=subdomain,
        guidance_text=guidance_text,
        language=language,
    )

    return ONE_SHOT_SYSTEM_PROMPT, user_prompt, applied_guidance


def validate_one_shot_response(response: str) -> tuple[bool, Optional[dict], str, str]:
    """
    Validate one-shot response format with full WritingBench criteria structure.

    Splits off the <think>…</think> reasoning trace (thinking-mode models like
    Qwen3) before validation and returns it so the caller can store it alongside
    the generated prompt.

    Returns:
        Tuple of (is_valid, parsed_json, thinking, failure_reason)
        failure_reason: a `FormatFailureReason` value — "ok" when valid;
        otherwise a FormatValidator reason or one of this layer's stricter
        one-shot schema checks. generate_prompts_batch aggregates these into
        generation_log["failure_reason_counts"] per run.
    """
    thinking, answer = split_thinking(response)

    score, parsed, fmt_reason = FormatValidator.validate_response(answer)
    if score != 1:
        return False, None, thinking, fmt_reason

    if "query" not in parsed or "criteria" not in parsed:
        logger.warning("Missing required fields: query or criteria")
        return False, None, thinking, FormatFailureReason.MISSING_QUERY_OR_CRITERIA.value

    criteria = parsed.get("criteria", [])
    if not isinstance(criteria, list) or len(criteria) == 0:
        logger.warning("Criteria is not a non-empty list")
        return False, None, thinking, FormatFailureReason.CRITERIA_NOT_LIST.value

    required_score_levels = ["1-2", "3-4", "5-6", "7-8", "9-10"]
    for criterion in criteria:
        if not isinstance(criterion, dict):
            logger.warning("Criterion is not a dict")
            return False, None, thinking, FormatFailureReason.CRITERION_NOT_DICT.value

        if "name" not in criterion or "criteria_description" not in criterion:
            logger.warning("Criterion missing name or criteria_description")
            return False, None, thinking, FormatFailureReason.CRITERION_MISSING_FIELDS.value

        for level in required_score_levels:
            if level not in criterion:
                logger.warning(f"Criterion missing score level: {level}")
                return False, None, thinking, FormatFailureReason.CRITERION_MISSING_SCORE_LEVEL.value

    return True, parsed, thinking, FormatFailureReason.OK.value


def generate_prompts_batch(
    model,
    tokenizer,
    num_prompts: int,
    num_format_retries: int = 3,
    seed: int = 42,
) -> PromptBatch:
    """
    Generate a batch of writing prompts using one-shot approach with full WritingBench criteria.

    Args:
        model: vLLM LLM instance
        tokenizer: HF tokenizer
        num_prompts: Number of prompts to generate
        num_format_retries: Max retries if format validation fails
        seed: Random seed

    Returns:
        PromptBatch with all generated prompts
    """
    import vllm

    sampler = DomainSampler(seed=seed)
    batch_id = f"one_shot_batch_{seed}_{int(time.time())}"
    batch = PromptBatch(batch_id=batch_id)

    prompt_counter = 0
    # Safety cap: avoid infinite loops if the model is consistently broken.
    max_total_attempts = num_prompts * (num_format_retries + 2)
    total_attempts = 0

    while len(batch.prompts) < num_prompts:
        if total_attempts >= max_total_attempts:
            print(
                f"  [WARN] Reached attempt cap ({max_total_attempts}); "
                f"collected {len(batch.prompts)}/{num_prompts} valid prompts."
            )
            break

        # Sample domain and subdomain
        domain_key, subdomain = sampler.sample_domain_subdomain_pair()
        domain_name = WRITING_DOMAINS[domain_key]['name']

        # Build prompt with applied guidance
        system_prompt, user_prompt, applied_guidance = build_one_shot_prompt(domain_key, subdomain)

        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
            prompt = f"system: {system_prompt}\n\nuser: {user_prompt}"

        # Generate with retries for this slot
        attempt = 0
        format_valid = False
        parsed_json = None
        thinking_trace = ""

        while attempt < num_format_retries and not format_valid:
            total_attempts += 1
            # Qwen3 thinking-mode best practices (no greedy decoding). max_tokens
            # is generous because the reasoning trace precedes the JSON output;
            # too small a budget truncates the fenced JSON block and fails validation.
            sampling_params = vllm.SamplingParams(
                max_tokens=32768,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                n=1,
                stop_token_ids=[tokenizer.eos_token_id],
            )

            completions = model.generate([prompt], sampling_params=sampling_params)
            response = completions[0].outputs[0].text

            is_valid, parsed_json, thinking_trace, fail_reason = validate_one_shot_response(response)

            if is_valid:
                format_valid = True
            else:
                counts = batch.generation_log.setdefault("failure_reason_counts", {})
                counts[fail_reason] = counts.get(fail_reason, 0) + 1
                if fail_reason == FormatFailureReason.NON_ENGLISH_QUERY.value:
                    # validate_one_shot_response now rejects non-English queries
                    # itself (via FormatValidator); keep the legacy counter fed
                    # so existing generation_log consumers see the same field.
                    batch.generation_log["language_filter_failures"] += 1
                batch.failures.append({
                    "slot": len(batch.prompts) + 1,
                    "attempt": attempt + 1,
                    "domain": domain_key,
                    "subdomain": subdomain,
                    "failure_reason": fail_reason,
                    "thinking_chars": len(thinking_trace),
                    "raw_response": response,
                })
                attempt += 1
                if attempt < num_format_retries:
                    print(f"  [{len(batch.prompts) + 1}/{num_prompts}] Validation failed ({fail_reason}), retrying...")

        if format_valid:
            prompt_counter += 1
            prompt_id = f"prompt_{prompt_counter:04d}"

            requirements_data = parsed_json.get("requirements", {})
            requirements = QueryRequirements(
                style=requirements_data.get("style"),
                format=requirements_data.get("format"),
                length=requirements_data.get("length"),
            )

            writing_prompt = WritingPrompt(
                prompt_id=prompt_id,
                domain=domain_key,
                domain_name=domain_name,
                subdomain=subdomain,
                query=parsed_json['query'],
                criteria=parsed_json.get('criteria', []),
                requirements=requirements,
                guidance_applied=applied_guidance,
                format_score=1,
                seed=seed,
                thinking=thinking_trace,
            )

            batch.add_prompt(writing_prompt)
            print(
                f"  [{len(batch.prompts)}/{num_prompts}] ✓ {domain_key}/{subdomain} "
                f"| criteria: {len(writing_prompt.criteria)} | guidance: {len(applied_guidance)} "
                f"| thinking: {len(thinking_trace)} chars"
            )
        else:
            batch.generation_log["format_validation_failures"] += 1
            print(
                f"  [{len(batch.prompts) + 1}/{num_prompts}] ✗ Format failed after "
                f"{num_format_retries} attempts — sampling new slot"
            )
            # Don't call batch.add_prompt(None); just loop to try a fresh slot.

    return batch


# =============================================================================
# Model loading + orchestration (library function; raises, never sys.exit —
# called both by the CLI below and by generate_prompts()'s pipeline wrapper)
# =============================================================================

def run_generation(
    *,
    model: str,
    num_samples: int,
    seed: int,
    out_path: str | Path,
    save_name: str = "generated_prompts",
    suffix: str = "run",
    format_retries: int = 3,
) -> PromptBatch:
    """Load a vLLM engine from `model`, generate `num_samples` WritingPrompts,
    write the batch to `out_path`, and return the batch.

    `save_name`/`suffix` are only used to name the W&B run; the caller
    decides `out_path` (RunPaths for the pipeline, or the CLI's own
    STORAGE_PATH-relative default below) so this function doesn't duplicate
    RunPaths' layout logic.
    """
    import vllm
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        raise PromptGenerationError(f"failed to load tokenizer for {model}: {e}") from e

    try:
        llm = vllm.LLM(model=model, tokenizer=model, seed=seed)
    except Exception as e:
        raise PromptGenerationError(f"failed to load vLLM model {model}: {e}") from e

    _coevolve_iter = int(os.getenv("COEVOLVE_ITERATION", "0"))
    _wandb_parent_run_id = os.getenv("WANDB_RUN_ID")
    _wandb_owned = False  # whether we created the run (and should finish it)

    if _WANDB_AVAILABLE and os.getenv("WANDB_MODE") != "disabled":
        if _wandb_parent_run_id:
            # Attach to the coevolve parent run created by the orchestrator.
            _wandb.init(
                project=os.getenv("WANDB_PROJECT", "r-zero"),
                id=_wandb_parent_run_id,
                resume="allow",
            )
        else:
            # Standalone invocation — create our own run.
            _wandb.init(
                project=os.getenv("WANDB_PROJECT", "r-zero"),
                job_type="prompt-generation",
                name=f"prompt-gen_{save_name}_{suffix}_iter{_coevolve_iter}",
                config={
                    "model": model,
                    "num_samples": num_samples,
                    "seed": seed,
                    "format_retries": format_retries,
                    "coevolve_iteration": _coevolve_iter,
                },
            )
            _wandb_owned = True

    print(f"\nGenerating {num_samples} writing prompts")
    print(f"Model: {model}\n")

    batch = generate_prompts_batch(
        model=llm,
        tokenizer=tokenizer,
        num_prompts=num_samples,
        num_format_retries=format_retries,
        seed=seed,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(batch.to_json())

    # Failed attempts (with full raw responses) go to a sidecar JSONL next to
    # the prompts JSON — kept out of batch.to_json() so the prompts file stays
    # small enough to eyeball, while nothing about a failure is lost.
    failures_path = out_path.with_suffix(".failures.jsonl")
    if batch.failures:
        with open(failures_path, "w") as f:
            f.write("\n".join(json.dumps(r) for r in batch.failures) + "\n")

    log = batch.generation_log
    total_criteria = sum(len(p.criteria) for p in batch.prompts)
    avg_criteria = total_criteria / len(batch.prompts) if batch.prompts else 0
    total_guidance = sum(len(p.guidance_applied) for p in batch.prompts)
    avg_guidance = total_guidance / len(batch.prompts) if batch.prompts else 0

    print(f"\n{'=' * 80}")
    print("One-Shot WritingBench Generation Complete")
    print(f"{'=' * 80}")
    print("\nGeneration Stats:")
    print(f"  Total generated: {log['total_generated']}/{log['total_attempted']}")
    print(f"  Skipped: {log['skipped']}")
    print(f"  Format validation failures: {log['format_validation_failures']}")
    print(f"  Language filter failures:   {log['language_filter_failures']}")
    reason_counts = log.get("failure_reason_counts", {})
    if reason_counts:
        print("  Failed attempts by reason:")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {reason}: {count}")
    print(f"\nOutput:")
    print(f"  File: {out_path}")
    if batch.failures:
        print(f"  Failures: {failures_path}")

    if _wandb_active():
        metrics = {
            "prompt_gen/total_generated": log["total_generated"],
            "prompt_gen/total_attempted": log["total_attempted"],
            "prompt_gen/skipped": log["skipped"],
            "prompt_gen/format_validation_failures": log["format_validation_failures"],
            "prompt_gen/success_rate": log["total_generated"] / max(log["total_attempted"], 1),
            "prompt_gen/language_filter_failures": log["language_filter_failures"],
            "prompt_gen/avg_criteria_per_prompt": avg_criteria,
            "prompt_gen/avg_guidance_per_prompt": avg_guidance,
            "prompt_gen/unique_domains": len(batch.domains_sampled),
            "prompt_gen/unique_subdomains": len(batch.subdomains_sampled),
        }
        for reason, count in reason_counts.items():
            metrics[f"prompt_gen/failure_reason/{reason}"] = count
        if batch.prompts:
            table = _wandb.Table(columns=["prompt_id", "domain", "subdomain", "query_preview", "num_criteria", "num_guidance"])
            for p in batch.prompts:
                table.add_data(p.prompt_id, p.domain, p.subdomain, p.query[:150], len(p.criteria), len(p.guidance_applied))
            metrics["prompt_gen/prompts"] = table
        if batch.failures:
            fail_table = _wandb.Table(
                columns=["slot", "attempt", "domain", "subdomain", "failure_reason", "thinking_chars", "response_preview"]
            )
            for r in batch.failures:
                # Preview only — the full raw_response lives in failures_path.
                fail_table.add_data(
                    r["slot"], r["attempt"], r["domain"], r["subdomain"],
                    r["failure_reason"], r["thinking_chars"], r["raw_response"][:600],
                )
            metrics["prompt_gen/failures"] = fail_table
        _wandb.log(metrics, step=_coevolve_iter)
        if _wandb_owned:
            _wandb.finish()

    return batch


# =============================================================================
# Pipeline step — called by the (future) orchestrator with a typed config
# =============================================================================

def generate_prompts(
    cfg: ExperimentConfig,
    paths: RunPaths,
    challenger_ckpt: str | Path,
    *,
    run_generation_fn: Callable[..., PromptBatch] = run_generation,
) -> Path:
    """Generate this iteration's solver training prompts from `challenger_ckpt`.

    Returns the path to the written prompts JSON (`paths.prompts_json`).
    Raises `PromptGenerationError` if model loading fails, or if fewer
    prompts were produced than `cfg.solver.num_train + cfg.solver.num_val`
    requires — with the generation_log context needed to see why, instead
    of a bare shortfall surfacing an hour later as "not enough rows" from
    `build_parquet`. `run_generation_fn` is injectable for testing.
    """
    num_prompts = cfg.solver.num_train + cfg.solver.num_val
    out_path = paths.prompts_json(suffix=cfg.run.profile)
    save_name = f"{paths.iter_abbr}_solver_prompts"

    batch = run_generation_fn(
        model=str(challenger_ckpt),
        num_samples=num_prompts,
        seed=cfg.run.seed,
        out_path=out_path,
        save_name=save_name,
        suffix=cfg.run.profile,
    )

    got = len(batch.prompts)
    if got < num_prompts:
        raise PromptGenerationError(
            f"expected >= {num_prompts} prompts, got {got} "
            f"(generation_log={batch.generation_log}, see {out_path})"
        )

    return out_path


# =============================================================================
# CLI — standalone invocation (used by scripts/creative_solver_smoke.sh
# during the Phase 2 transition; the pipeline calls generate_prompts() above
# directly instead).
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-shot WritingBench prompt generation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="Model to use")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of prompts to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--suffix", type=str, default="test", help="Suffix for output file")
    parser.add_argument("--save_name", type=str, default="one_shot_writingbench", help="Base name for output")
    parser.add_argument("--format_retries", type=int, default=3, help="Format validation retries")

    args = parser.parse_args()

    storage_path = os.getenv("STORAGE_PATH")
    if not storage_path:
        print("ERROR: STORAGE_PATH is not set", file=sys.stderr)
        sys.exit(1)
    cli_out_path = Path(storage_path) / "generated_questions_one_shot" / f"{args.save_name}_{args.suffix}.json"

    try:
        run_generation(
            model=args.model,
            num_samples=args.num_samples,
            seed=args.seed,
            out_path=cli_out_path,
            save_name=args.save_name,
            suffix=args.suffix,
            format_retries=args.format_retries,
        )
    except Exception as e:
        print(f"[{args.suffix}] generate_prompts failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
