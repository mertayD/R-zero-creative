"""steps/build_parquet.py — prompts -> VERL train/val parquet.

Port of the two bash heredocs this replaces:
  * `creative_challenger_smoke.sh` Step 0 -> `build_challenger_parquet`
  * `creative_solver_smoke.sh` Step 1     -> `build_solver_parquet`

Row schemas are unchanged from those heredocs — parquet consumers (the two
reward callers, and the parity diff in T2.9) see identical columns, just
produced by a pure, testable function instead of Python pasted into a bash
string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd

from creative_rzero.config import ExperimentConfig
from creative_rzero.data.writing_prompt import WritingPrompt
from creative_rzero.paths import RunPaths
from creative_rzero.steps.generate_prompts import DomainSampler, build_one_shot_prompt
from question_generate.creative_writing_prompts import WRITING_DOMAINS


def build_challenger_parquet(cfg: ExperimentConfig, paths: RunPaths) -> Tuple[Path, Path]:
    """Sample (domain, subdomain) pairs and build the challenger train/val parquet.

    Row schema (unchanged from creative_challenger_smoke.sh Step 0): problem,
    answer, domain, domain_name, subdomain, seed, iteration, run_id,
    challenger_init_model, solver_oracle_model. The challenger has no
    reference answer, so `answer` carries a JSON metadata payload instead
    (same convention as the solver's WritingPrompt-in-answer): verl forwards
    it to the reward function as ground_truth, where it feeds the per-rollout
    log — it never affects the reward itself.
    """
    sampler = DomainSampler(seed=cfg.run.seed)

    def make_row() -> dict:
        domain_key, subdomain = sampler.sample_domain_subdomain_pair()
        domain_name = WRITING_DOMAINS[domain_key].get("name", domain_key)
        _, user_prompt, _ = build_one_shot_prompt(domain_key, subdomain)
        return {
            "problem": user_prompt,
            "answer": json.dumps({
                "domain": domain_key,
                "domain_name": domain_name,
                "subdomain": subdomain,
                "problem": user_prompt,
            }),
            "domain": domain_key,
            "domain_name": domain_name,
            "subdomain": subdomain,
            "seed": cfg.run.seed,
            "iteration": paths.iteration,
            "run_id": paths.run_ts,
            "challenger_init_model": cfg.challenger.model_path,
            "solver_oracle_model": cfg.solver.model_path,
        }

    train_rows = [make_row() for _ in range(cfg.challenger.num_train)]
    val_rows = [make_row() for _ in range(cfg.challenger.num_val)]

    train_path = paths.train_parquet("challenger")
    val_path = paths.val_parquet("challenger")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    return train_path, val_path


def build_solver_parquet(
    prompts_json: Path,
    challenger_checkpoint: str,
    cfg: ExperimentConfig,
    paths: RunPaths,
) -> Tuple[Path, Path]:
    """Deserialize challenger-generated WritingPrompts and build the solver
    train/val parquet.

    Row schema (unchanged from creative_solver_smoke.sh Step 1): problem,
    answer, prompt_id, domain, domain_name, subdomain, num_criteria,
    criteria_json, requirements_style, requirements_format,
    requirements_length, guidance_applied_json, num_guidance_applied,
    format_score, language, seed, iteration, run_id, challenger_checkpoint.

    Only rows with a non-empty query and >=1 criterion are kept; bad rows are
    skip-logged (printed), matching the heredoc's behavior — everything
    downstream trusts the parquet unconditionally rather than re-validating.
    Raises `ValueError` (not a silent truncated dataset) if fewer than
    `cfg.solver.num_train + cfg.solver.num_val` rows survive validation.

    `challenger_checkpoint` is the merged checkpoint path actually used to
    generate `prompts_json` (steps/generate_prompts.py's return value's
    input, not `cfg.challenger.model_path` — the two diverge after the first
    co-evolution iteration).
    """
    data = json.loads(Path(prompts_json).read_text())
    raw_prompts = data.get("prompts", [])

    valid: list[WritingPrompt] = []
    skipped = 0
    for p in raw_prompts:
        try:
            wp = WritingPrompt.from_dict(p)
            assert wp.query.strip(), f"empty query in {wp.prompt_id}"
            assert wp.criteria, f"no criteria in {wp.prompt_id}"
            valid.append(wp)
        except Exception as e:
            print(f"  [skip] {e}")
            skipped += 1

    num_train = cfg.solver.num_train
    num_val = cfg.solver.num_val
    if len(valid) < num_train + num_val:
        raise ValueError(
            f"need {num_train + num_val} valid prompts, got {len(valid)} (skipped {skipped})"
        )

    train_prompts = valid[:num_train]
    val_prompts = valid[num_train : num_train + num_val]

    def make_row(wp: WritingPrompt) -> dict:
        req = wp.requirements.to_dict() if wp.requirements else {}
        return {
            "problem": wp.query,
            "answer": json.dumps(wp.to_dict()),
            "prompt_id": wp.prompt_id,
            "domain": wp.domain,
            "domain_name": wp.domain_name,
            "subdomain": wp.subdomain,
            "num_criteria": len(wp.criteria),
            "criteria_json": json.dumps(wp.criteria),
            "requirements_style": req.get("style") or "",
            "requirements_format": req.get("format") or "",
            "requirements_length": req.get("length") or "",
            "guidance_applied_json": json.dumps(wp.guidance_applied),
            "num_guidance_applied": len(wp.guidance_applied),
            "format_score": wp.format_score,
            "language": wp.language,
            "seed": wp.seed,
            "iteration": paths.iteration,
            "run_id": paths.run_ts,
            "challenger_checkpoint": challenger_checkpoint,
        }

    train_path = paths.train_parquet("solver")
    val_path = paths.val_parquet("solver")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([make_row(wp) for wp in train_prompts]).to_parquet(train_path, index=False)
    pd.DataFrame([make_row(wp) for wp in val_prompts]).to_parquet(val_path, index=False)
    return train_path, val_path
