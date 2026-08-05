"""RunPaths — the single source of truth for run/checkpoint/parquet/log layout.

Today this layout is assembled independently in `modal_run.py`, the two
`scripts/creative_{challenger,solver}_smoke.sh` scripts, and both reward
callers — three-way string duplication that produced the A-table path
mismatch (REFACTOR_PLAN.md class F). `steps/`, `orchestrator.py`, and
`rewards/` (T2.4 onward) import `RunPaths` instead of building f-strings;
one method = one string, defined once, so path-mismatch bugs become a
type-level impossibility.

`run_ts` is always supplied by the caller (the orchestrator, or a
standalone launcher generating its own timestamp up front) instead of the
old RUN_ID-set-vs-unset branch in the bash scripts — Phase 2 kills that
env-var contract (REFACTOR_PLAN.md class A) entirely; RunPaths is the
replacement.
"""

from __future__ import annotations

from pathlib import Path

# The two trainable models in the co-evolution loop. Every method that takes
# a `role` argument validates against this tuple so a typo'd role fails at
# the call site instead of silently producing a path nothing writes to.
VALID_ROLES = ("challenger", "solver")


def _check_role(role: str) -> None:
    """Shared validation for every `role`-taking method below."""
    if role not in VALID_ROLES:
        raise ValueError(f"role={role!r} must be one of {VALID_ROLES}")


class RunPaths:
    """Path authority for one co-evolution run.

    `abbr` is the short project name for the whole run, stable across
    iterations (what T1.3 called the coevolve-level `abbr` — e.g.
    `qwen3-4b-coevolve`). `iter_abbr` (below) folds in `run_ts` and
    `iteration` to name one specific challenger/solver training pass within
    that run; `SAVE_NAME` and the parquet/checkpoint layout are keyed on
    `iter_abbr`, not the bare `abbr`.

    A few paths are deliberately keyed on `abbr` alone rather than
    `iter_abbr` because they're run-spanning, not iteration-spanning:
    `state_file()` (one resume file per run) and `manifest()`/`run_dir()`
    (one manifest per run). Everything else — checkpoints, parquet, prompts,
    rollout logs — is per-iteration.
    """

    def __init__(self, storage: str | Path, abbr: str, run_ts: str, iteration: int = 1):
        self.storage = Path(storage)  # volume root, e.g. Modal's REMOTE_STORAGE_PATH
        self.abbr = abbr  # run-level short name, constant across iterations
        self.run_ts = run_ts  # run-level timestamp, constant across iterations
        self.iteration = iteration  # current co-evolution iteration (1-indexed)

    # ---- naming -------------------------------------------------------
    # Every other path method is built from `iter_abbr` or `save_name`, so
    # get these two right and the rest follow.
    @property
    def iter_abbr(self) -> str:
        """Per-iteration run name, e.g. `qwen3-4b-coevolve_20260805_120000_iter1`."""
        return f"{self.abbr}_{self.run_ts}_iter{self.iteration}"

    def save_name(self, role: str) -> str:
        """VERL `trainer.experiment_name` / checkpoint directory name for `role`.

        The `_v1` suffix mirrors the bash scripts' SAVE_NAME convention
        (originally a hedge against re-running the same abbr/timestamp
        combination); kept as-is so old and new checkpoint names read the
        same way during the Phase 2 migration.
        """
        _check_role(role)
        return f"{self.iter_abbr}_{role}_v1"

    # ---- checkpoints ----------------------------------------------------
    def checkpoint_root(self, role: str) -> Path:
        """VERL `trainer.save_checkpoint_path` for `role` — the directory VERL
        writes `global_step_N/` subdirectories under during training."""
        return self.storage / "models" / self.save_name(role)

    def checkpoint_step_dir(self, role: str, step: int) -> Path:
        """Pre-merge actor directory for a global step — `model_merger.py`'s
        `--local_dir` input. Contains the sharded `model_world_size_*_rank_*.pt`
        files VERL saves, not yet consolidated into a HuggingFace directory."""
        return self.checkpoint_root(role) / f"global_step_{step}" / "actor"

    def merged_checkpoint(self, role: str, step: int) -> Path:
        """Merged HuggingFace directory for a global step.

        `model_merger.py` always writes its output to `<local_dir>/huggingface`
        (it asserts `local_dir` doesn't already end in `huggingface`), so this
        is simply `checkpoint_step_dir(role, step) / "huggingface"`.
        """
        return self.checkpoint_step_dir(role, step) / "huggingface"

    def challenger_ckpt(self, step: int) -> Path:
        """Convenience alias for `merged_checkpoint("challenger", step)`."""
        return self.merged_checkpoint("challenger", step)

    def solver_ckpt(self, step: int) -> Path:
        """Convenience alias for `merged_checkpoint("solver", step)`."""
        return self.merged_checkpoint("solver", step)

    def last_ckpt_file(self, role: str) -> Path:
        """Sidecar file a training step writes its merged ckpt path to, so the
        orchestrator reads it back instead of reconstructing SAVE_NAME/global_step
        independently (the source of the T1.3 path-mismatch bug).

        Keyed on `iter_abbr`, not `abbr` — the training step only knows the
        per-iteration name it was invoked with.
        """
        _check_role(role)
        return self.storage / "models" / f"{self.iter_abbr}_{role}_last_ckpt.txt"

    # ---- data -------------------------------------------------------------
    def prompts_json(self, suffix: str = "smoke") -> Path:
        """Challenger-generated WritingPrompt batch used to build the solver's
        train/val parquet. `suffix` mirrors the old script's `--suffix` arg
        (e.g. run-size tag); default matches the current smoke-run convention."""
        return self.storage / "generated_questions_one_shot" / f"{self.iter_abbr}_solver_prompts_{suffix}.json"

    def train_parquet(self, role: str) -> Path:
        """VERL `data.train_files` for `role`."""
        return self._parquet(role, "train")

    def val_parquet(self, role: str) -> Path:
        """VERL `data.val_files` for `role`."""
        return self._parquet(role, "val")

    def _parquet(self, role: str, split: str) -> Path:
        # Challenger parquet has no role infix (`{iter_abbr}_train.parquet`);
        # solver parquet does (`{iter_abbr}_solver_train.parquet`) — this
        # asymmetry comes straight from the two bash scripts' existing
        # filenames and is preserved here rather than "fixed", since
        # `creative_smoke/` already has files on disk under the old names.
        _check_role(role)
        infix = "" if role == "challenger" else "solver_"
        return self.storage / "creative_smoke" / f"{self.iter_abbr}_{infix}{split}.parquet"

    # ---- logs / state -------------------------------------------------------
    def rollout_log(self, role: str) -> Path:
        """Per-rollout JSONL reward log for `role` — what the reward callers
        append to and what `creative_{challenger,solver}_smoke.sh` Step 4
        uploads to W&B as a Table. Named after `save_name`, matching the old
        scripts' `VERL_EXPERIMENT_NAME`-keyed log file."""
        return self.storage / "reward_logs" / f"{self.save_name(role)}.jsonl"

    def judge_cache(self) -> Path:
        """On-disk judge response cache directory (T1.5) — keyed internally by
        the caller on `sha256(system+prompt)`, not by run, so it's shared
        across runs on the same volume rather than living under `run_dir()`."""
        return self.storage / "judge_cache"

    def state_file(self) -> Path:
        """Co-evolution orchestrator resume state — keyed on `abbr` alone, since
        one state file spans every iteration of a run, not just one (it's what
        lets a Modal retry resume mid-run rather than restarting iteration 1)."""
        return self.storage / "coevolve_state" / f"{self.abbr}.json"

    def run_dir(self) -> Path:
        """Root directory for run-spanning artifacts (currently just the T4.3
        manifest). Keyed on `abbr` + `run_ts`, not `iter_abbr` — one run_dir
        per co-evolution run, shared across all its iterations."""
        return self.storage / "runs" / f"{self.abbr}_{self.run_ts}"

    def manifest(self) -> Path:
        """Resolved-config + git-SHA + checkpoint-provenance record (T4.3)."""
        return self.run_dir() / "manifest.json"
