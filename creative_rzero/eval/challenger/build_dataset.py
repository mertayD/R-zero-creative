"""creative_rzero/eval/challenger/build_dataset.py — Phase 1: build the
frozen challenger eval set (every domain/subdomain pair x REPLICATES_PER_PAIR
guidance replicates) and push it to the HF Hub.

Standalone: run once (or whenever the eval set itself needs to change), not
per training run or per checkpoint. run_eval.py (Phase 2) pulls the
resulting Hub dataset by repo name so every checkpoint is scored against the
identical set of prompts — comparable results across runs, not eval-set
noise.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from creative_rzero.steps.generate_prompts import build_one_shot_prompt  # noqa: E402
from question_generate.creative_writing_prompts import WRITING_DOMAINS  # noqa: E402

load_dotenv()

REPLICATES_PER_PAIR = 10
DEFAULT_REPO_NAME = "challenger-eval-v1"


def build_rows(replicates_per_pair: int = REPLICATES_PER_PAIR) -> list[dict]:
    """Deterministically render every (domain, subdomain) pair x replicate.

    Iterates WRITING_DOMAINS/its subdomain lists in their fixed dict/list
    order (not DomainSampler's random sampling) so coverage is exhaustive
    rather than probabilistic. Before each build_one_shot_prompt() call,
    reseeds Python's global `random` state with a key derived from
    `(domain_key, subdomain, replicate_idx)` — build_one_shot_prompt's
    guidance-pool sampling reads that global state, so this is what makes
    the `replicates_per_pair` replicates for one pair independently sampled
    yet exactly reproducible across re-runs of this function.
    `random.seed(str)` hashes via sha512 (Python's `version=2` default), not
    the process-randomized builtin `hash()`, so this is stable across
    processes and interpreter restarts, not just within one run.
    """
    rows: list[dict] = []
    eval_id = 0
    for domain_key, domain in WRITING_DOMAINS.items():
        for subdomain in domain["subdomains"]:
            for replicate_idx in range(replicates_per_pair):
                random.seed(f"{domain_key}|{subdomain}|{replicate_idx}")
                system_prompt, user_prompt, applied_guidance = build_one_shot_prompt(domain_key, subdomain)
                rows.append(
                    {
                        "eval_id": eval_id,
                        "domain": domain_key,
                        "domain_name": domain["name"],
                        "subdomain": subdomain,
                        "replicate_idx": replicate_idx,
                        "guidance_applied": applied_guidance,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    }
                )
                eval_id += 1
    return rows


def write_jsonl(rows: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return out_path


def push_to_hub(
    rows: list[dict],
    repo_name: str,
    hf_token: str | None = None,
    hf_username: str | None = None,
) -> str:
    """Push `rows` as a private Hub dataset with a single "eval" split,
    mirroring question_evaluate/upload.py's Dataset.from_list ->
    DatasetDict -> push_to_hub pattern. Returns the pushed repo id."""
    from datasets import Dataset, DatasetDict
    from huggingface_hub import login

    hf_token = hf_token or os.getenv("HF_TOKEN")
    hf_username = hf_username or os.getenv("HUGGINGFACENAME")
    if not hf_token or not hf_username:
        raise RuntimeError(
            "HF_TOKEN and HUGGINGFACENAME must be set (env or --hf-token/--hf-username) "
            "to push the eval dataset to the Hub. Use --dry-run to skip the push."
        )
    login(token=hf_token)

    dataset = DatasetDict({"eval": Dataset.from_list(rows)})
    repo_id = f"{hf_username}/{repo_name}"
    dataset.push_to_hub(repo_id, private=True)
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the frozen challenger eval set")
    parser.add_argument("--repo_name", type=str, default=DEFAULT_REPO_NAME)
    parser.add_argument("--replicates_per_pair", type=int, default=REPLICATES_PER_PAIR)
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Local JSONL path; defaults to $STORAGE_PATH/eval/challenger/<repo_name>.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the local JSONL only, skip the Hub push")
    args = parser.parse_args()

    rows = build_rows(args.replicates_per_pair)

    default_out = Path(os.getenv("STORAGE_PATH", ".")) / "eval" / "challenger" / f"{args.repo_name}.jsonl"
    out_path = write_jsonl(rows, args.out_path or default_out)
    print(f"Wrote {len(rows)} rows to {out_path}")

    if args.dry_run:
        print("--dry-run set: skipping Hub push")
    else:
        repo_id = push_to_hub(rows, args.repo_name)
        print(f"Pushed {len(rows)} rows to hf://{repo_id} (split 'eval')")
