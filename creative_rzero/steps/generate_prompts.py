"""steps/generate_prompts.py — challenger checkpoint -> WritingPrompt JSON.

Port of `creative_solver_smoke.sh` Step 0. The heavy lifting (loading a vLLM
model, sampling, format-validating, writing the batch JSON) stays in
`question_generate/one_shot_creative_question_generate.py` — that module
loads vllm/transformers at import time, which is exactly what `config.py`'s
docstring says to keep out of the config-validation path. This step invokes
it as a subprocess with explicit, typed args (the alternative the REFACTOR
plan calls out to "calling the generator in-process"), so a fixed seed
produces byte-identical output to the old bash invocation: same script, same
args, same interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from creative_rzero.config import ExperimentConfig
from creative_rzero.paths import RunPaths

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATOR_SCRIPT = REPO_ROOT / "question_generate" / "one_shot_creative_question_generate.py"


class PromptGenerationError(RuntimeError):
    """Raised when the generator subprocess fails or under-delivers prompts."""


def generate_prompts(
    cfg: ExperimentConfig,
    paths: RunPaths,
    challenger_ckpt: str | Path,
    *,
    run: Optional[Callable[[list[str]], None]] = None,
) -> Path:
    """Generate this iteration's solver training prompts from `challenger_ckpt`.

    Returns the path to the written prompts JSON (`paths.prompts_json`,
    matching what the subprocess is told to write to via --save_name/--suffix
    so the two never drift apart). Raises `PromptGenerationError` if the
    subprocess fails, or if it produced fewer prompts than
    `cfg.solver.num_train + cfg.solver.num_val` requires — with the
    generation_log context needed to see why, instead of a bare shortfall
    surfacing an hour later as "not enough rows" from `build_parquet`.
    """
    num_prompts = cfg.solver.num_train + cfg.solver.num_val
    save_name = f"{paths.iter_abbr}_solver_prompts"
    out_path = paths.prompts_json(suffix=cfg.run.profile)

    cmd = [
        sys.executable,
        str(GENERATOR_SCRIPT),
        "--model", str(challenger_ckpt),
        "--num_samples", str(num_prompts),
        "--seed", str(cfg.run.seed),
        "--save_name", save_name,
        "--suffix", cfg.run.profile,
    ]

    runner = run or _run_subprocess
    runner(cmd)

    if not out_path.exists():
        raise PromptGenerationError(f"generator did not produce {out_path}")

    data = json.loads(out_path.read_text())
    got = len(data.get("prompts", []))
    if got < num_prompts:
        raise PromptGenerationError(
            f"expected >= {num_prompts} prompts, got {got} "
            f"(generation_log={data.get('generation_log')}, see {out_path})"
        )

    return out_path


def _run_subprocess(cmd: list[str]) -> None:
    import os

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "7")
    subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)
