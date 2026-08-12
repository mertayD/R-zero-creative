"""orchestrator.py — the co-evolution loop in pure Python (T2.8).

Port of `modal_run.py::train_creative_coevolve` (state/resume/verify/upload,
unchanged in behavior) plus the two creative bash scripts' non-verl steps:

  Phase A (challenger)  — was `creative_challenger_smoke.sh`:
      build challenger parquet -> start vLLM solver oracle on the aux GPU
      -> launch verl training -> merge checkpoint -> upload rollout log.
  Phase B (solver)      — was `creative_solver_smoke.sh`:
      generate prompts from the challenger checkpoint (subprocess, aux GPU)
      -> build solver parquet -> launch verl training -> merge checkpoint
      -> upload rollout log.

Modal-specific concerns (volume commit/reload, remote function dispatch)
stay in `modal_app.py` and are injected as callables, so this module is
importable and testable without Modal installed.

GPU layout: verl trains on GPUs `0..n_gpus_per_node-1`; the vLLM solver
oracle (Phase A) and prompt generation (Phase B) run on the *next* GPU
(`n_gpus_per_node`). The machine therefore needs `n_gpus_per_node + 1`
GPUs. This replaces the old scripts' hardcoded "GPUs 0-3 train, GPU 7
serves" layout, which silently wasted GPUs 4-6 and pinned every run to an
8-GPU box.

Prompt generation runs as a subprocess (`python -m
creative_rzero.steps.generate_prompts`), not via the in-process
`generate_prompts()` step: a vLLM engine's CUDA context is only reliably
released when its process exits, and the verl training subprocess launched
right afterwards needs that memory (REFACTOR_PLAN.md T2.4 follow-up #1 —
the isolation decision belongs here, at the orchestrator level).
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from creative_rzero.config import ExperimentConfig
from creative_rzero.paths import RunPaths
from creative_rzero.steps.build_parquet import build_challenger_parquet, build_solver_parquet
from creative_rzero.steps.merge_ckpt import merge_checkpoint
from creative_rzero.steps.report import report_generation, report_phase, summary_run_id
from creative_rzero.steps.train_verl import launch_training

REPO_ROOT = Path(__file__).resolve().parent.parent

SOLVER_SERVER_HEALTH_TIMEOUT_S = 600  # matches the old script's 60 x 10s poll
SOLVER_SERVER_POLL_INTERVAL_S = 10


class CheckpointVerificationError(RuntimeError):
    """Raised when a phase's merged checkpoint is missing config.json or weights."""


class PromptShortfallError(RuntimeError):
    """Raised when the prompt-generation subprocess produced fewer valid
    prompts than the solver phase needs."""


class SolverServerError(RuntimeError):
    """Raised when the vLLM solver oracle fails to become healthy."""


class SftCriticJudgeError(RuntimeError):
    """Raised when the deployed sft-critic judge service never becomes healthy."""


# ---------------------------------------------------------------------------
# GPU layout
# ---------------------------------------------------------------------------

def n_training_gpus(cfg: ExperimentConfig) -> int:
    return int(cfg.verl["trainer"]["n_gpus_per_node"])


def training_gpu_ids(cfg: ExperimentConfig) -> str:
    """CUDA_VISIBLE_DEVICES value for the verl training subprocess."""
    return ",".join(str(i) for i in range(n_training_gpus(cfg)))


def aux_gpu_id(cfg: ExperimentConfig) -> int:
    """The GPU the vLLM solver oracle / prompt generation uses — the first
    one verl doesn't train on."""
    return n_training_gpus(cfg)


# ---------------------------------------------------------------------------
# Resume state — same JSON schema as modal_run.py::train_creative_coevolve
# ---------------------------------------------------------------------------

@dataclass
class CoevolveState:
    run_ts: str
    wandb_group: str
    next_iter: int
    next_phase: str  # "challenger" | "solver"
    current_challenger: str
    current_solver: str

    def save(self, state_file: Path) -> None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, state_file: Path) -> Optional["CoevolveState"]:
        if not state_file.exists():
            return None
        return cls(**json.loads(state_file.read_text()))


# ---------------------------------------------------------------------------
# Shared checks / uploads
# ---------------------------------------------------------------------------

def verify_checkpoint(label: str, ckpt: str, reload_fn: Optional[Callable[[], None]] = None) -> None:
    """Fail fast if `ckpt` is not a merged HF directory with weights.
    `reload_fn` (e.g. Modal volume.reload) refreshes the view of a shared
    volume written to by another container before the existence check."""
    if reload_fn is not None:
        reload_fn()
    if not os.path.exists(os.path.join(ckpt, "config.json")):
        raise CheckpointVerificationError(f"{label}: config.json missing at {ckpt}")
    has_weights = bool(glob.glob(os.path.join(ckpt, "model*.safetensors"))) or bool(
        glob.glob(os.path.join(ckpt, "pytorch_model*.bin"))
    )
    if not has_weights:
        raise CheckpointVerificationError(f"{label}: no weight files at {ckpt}")
    print(f"[OK] {label} checkpoint verified: {ckpt}")


def upload_parquets_to_hf(paths: RunPaths, role: str) -> None:
    """Upload a phase's train/val parquet to the HF datasets repo
    (port of modal_run.py::_upload_parquets). Best-effort."""
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_name = os.environ.get("HUGGINGFACENAME", "")
    if not hf_token or not hf_name:
        print("    [HF] HF_TOKEN or HUGGINGFACENAME not set — skipping upload")
        return
    repo_id = f"{hf_name}/r-zero-creative-datasets"
    hf_prefix = f"{paths.abbr}/{paths.run_ts}/iter{paths.iteration}/{role}"
    print(f"    [HF] -> {repo_id}/{hf_prefix}/")
    for local, hf_path in [
        (paths.train_parquet(role), f"{hf_prefix}/train.parquet"),
        (paths.val_parquet(role), f"{hf_prefix}/val.parquet"),
    ]:
        try:
            subprocess.run(
                ["huggingface-cli", "upload", repo_id, str(local), hf_path,
                 "--repo-type", "dataset", "--token", hf_token],
                check=True,
            )
        except Exception as e:
            print(f"    [HF] upload failed {hf_path}: {e}")


# ---------------------------------------------------------------------------
# vLLM solver oracle (Phase A reward dependency)
# ---------------------------------------------------------------------------

@contextmanager
def solver_server(
    model: str,
    port: int,
    gpu_id: int,
    *,
    health_timeout_s: int = SOLVER_SERVER_HEALTH_TIMEOUT_S,
    poll_interval_s: int = SOLVER_SERVER_POLL_INTERVAL_S,
) -> Iterator["subprocess.Popen[bytes]"]:
    """Run vLLM's OpenAI-compatible server on `gpu_id` for the duration of
    the `with` block, waiting for /health before yielding (was bash Steps 1/5)."""
    import requests

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", model, "--port", str(port), "--dtype", "auto", "--disable-log-requests"],
        env=env,
        cwd=REPO_ROOT,
    )
    print(f"[solver_server] vLLM solver PID={proc.pid} model={model} gpu={gpu_id} port={port}")
    try:
        deadline = time.monotonic() + health_timeout_s
        while True:
            if proc.poll() is not None:
                raise SolverServerError(f"vLLM solver server exited early with code {proc.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=5).ok:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise SolverServerError(f"vLLM solver server not ready after {health_timeout_s}s")
            print("[solver_server] ... waiting for vLLM solver")
            time.sleep(poll_interval_s)
        print("[solver_server] vLLM solver server ready.")
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("[solver_server] vLLM solver stopped.")


# ---------------------------------------------------------------------------
# sft-critic judge warm-up (Phase A + Phase B reward dependency, T3.11)
# ---------------------------------------------------------------------------

SFT_CRITIC_HEALTH_TIMEOUT_S = 300
SFT_CRITIC_POLL_INTERVAL_S = 10


def wait_for_sft_critic_judge(
    url: str,
    *,
    health_timeout_s: int = SFT_CRITIC_HEALTH_TIMEOUT_S,
    poll_interval_s: int = SFT_CRITIC_POLL_INTERVAL_S,
) -> None:
    """Warm up the persistent, scale-to-zero sft-critic judge service
    (modal_app.py::critic_judge_server, REFACTOR_PLAN.md §6.2) before
    training starts, so a cold vLLM boot (roughly 1-3 min for a 7B model)
    lands here rather than stalling the first judge call mid-training.

    Unlike solver_server(), this doesn't spawn or own a process — the
    service is already deployed and scales itself (min_containers=0); this
    only polls its /health endpoint until a container is up."""
    import requests

    print(f"[sft_critic_judge] warming up {url} ...")
    deadline = time.monotonic() + health_timeout_s
    while True:
        try:
            if requests.get(f"{url.rstrip('/')}/health", timeout=5).ok:
                break
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise SftCriticJudgeError(f"sft-critic judge at {url} not ready after {health_timeout_s}s")
        print("[sft_critic_judge] ... waiting for sft-critic judge")
        time.sleep(poll_interval_s)
    print("[sft_critic_judge] sft-critic judge ready.")


# ---------------------------------------------------------------------------
# Phase B Step 0 — prompt generation (subprocess for GPU-memory isolation)
# ---------------------------------------------------------------------------

def generate_solver_prompts(
    cfg: ExperimentConfig,
    paths: RunPaths,
    challenger_ckpt: str,
    *,
    run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run,
) -> Path:
    """Generate this iteration's solver prompts from `challenger_ckpt` on the
    aux GPU, in a child process so the vLLM engine's GPU memory is fully
    released before verl training starts (see module docstring). Returns the
    prompts JSON path after checking the count covers train + val."""
    needed = cfg.solver.num_train + cfg.solver.num_val
    out_path = paths.prompts_json(suffix=cfg.run.profile)

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(aux_gpu_id(cfg)),
        "STORAGE_PATH": str(paths.storage),  # the CLI derives its output path from this
        "COEVOLVE_ITERATION": str(paths.iteration),
    }
    cmd = [
        sys.executable, "-m", "creative_rzero.steps.generate_prompts",
        "--model", str(challenger_ckpt),
        "--num_samples", str(needed),
        "--seed", str(cfg.run.seed),
        "--save_name", f"{paths.iter_abbr}_solver_prompts",
        "--suffix", cfg.run.profile,
    ]
    result = run(cmd, env=env, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise PromptShortfallError(
            f"prompt generation subprocess exited {result.returncode} for {challenger_ckpt}"
        )

    batch = json.loads(out_path.read_text())
    got = len(batch.get("prompts", []))
    if got < needed:
        raise PromptShortfallError(
            f"expected >= {needed} prompts, got {got} "
            f"(generation_log={batch.get('generation_log')}, see {out_path})"
        )
    return out_path


# ---------------------------------------------------------------------------
# The two training phases (each runs inside one GPU container)
# ---------------------------------------------------------------------------

def run_challenger_phase(cfg: ExperimentConfig, paths: RunPaths) -> str:
    """Phase A: train the challenger against the frozen solver oracle.
    `cfg.challenger.model_path` is the model being trained;
    `cfg.solver.model_path` is the frozen oracle served over HTTP.
    Returns the merged HF checkpoint path."""
    run_dir = paths.run_dir()
    print(f">>> Phase A — challenger training ({cfg.challenger.model_path}, "
          f"{cfg.challenger.max_steps} steps, iter {paths.iteration})")

    build_challenger_parquet(cfg, paths)

    os.environ["CUDA_VISIBLE_DEVICES"] = training_gpu_ids(cfg)  # forwarded by launch_training
    with solver_server(cfg.solver.model_path, cfg.challenger.solver_query.port, aux_gpu_id(cfg)):
        launch_training(cfg, paths, "challenger", run_dir)

    # Report before merging: merge raises on a missing/collapsed checkpoint,
    # and the rollout tables are exactly what you need to debug that failure.
    report_phase(paths, "challenger", max_steps=cfg.challenger.max_steps)
    merged = merge_checkpoint(paths, "challenger", max_steps=cfg.challenger.max_steps)
    return str(merged)


def run_solver_phase(cfg: ExperimentConfig, paths: RunPaths, challenger_ckpt: str) -> str:
    """Phase B: train the solver on prompts generated by `challenger_ckpt`.
    `cfg.solver.model_path` is the model being trained. Returns the merged
    HF checkpoint path."""
    run_dir = paths.run_dir()
    print(f">>> Phase B — solver training ({cfg.solver.model_path}, "
          f"{cfg.solver.max_steps} steps, G={cfg.solver.rollout_n}, iter {paths.iteration})")

    try:
        prompts_json = generate_solver_prompts(cfg, paths, challenger_ckpt)
    except PromptShortfallError:
        # A shortfall is the single most common Phase B failure — report the
        # generation artifacts before propagating, so the run's W&B page
        # explains the crash instead of just ending at it.
        report_generation(paths, paths.prompts_json(suffix=cfg.run.profile))
        raise
    report_generation(paths, prompts_json)
    build_solver_parquet(prompts_json, challenger_ckpt, cfg, paths)

    os.environ["CUDA_VISIBLE_DEVICES"] = training_gpu_ids(cfg)
    launch_training(cfg, paths, "solver", run_dir)

    report_phase(paths, "solver", max_steps=cfg.solver.max_steps)
    merged = merge_checkpoint(paths, "solver", max_steps=cfg.solver.max_steps)
    return str(merged)


# ---------------------------------------------------------------------------
# The co-evolution loop (runs in the CPU orchestrator container)
# ---------------------------------------------------------------------------

def _wandb_summary_run(cfg: ExperimentConfig, paths: RunPaths, wandb_group: str) -> None:
    """Create the W&B group's summary run so sub-runs nest under it.

    Uses `report.summary_run_id` so the phases' `report_phase` calls resume
    *this* run rather than creating their own — one run holds the config plus
    every health table for the whole co-evolution.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return
    try:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "r-zero-creative"),
            id=summary_run_id(paths),
            resume="allow",
            job_type="coevolve_summary",
            name=f"coevolve_{cfg.run.abbr}_{paths.run_ts}",
            group=wandb_group,
            config={
                "challenger_model": cfg.challenger.model_path,
                "solver_model": cfg.solver.model_path,
                "num_iters": cfg.run.num_iters,
                "seed": cfg.run.seed,
                "profile": cfg.run.profile,
                "run_ts": paths.run_ts,
            },
            tags=["coevolve"],
        )
        run.finish()
        print(f"[W&B] Group: {wandb_group}")
    except Exception as e:
        print(f"[W&B] Summary run skipped: {e}")


def run_coevolve(
    cfg: ExperimentConfig,
    storage: str | Path,
    *,
    train_challenger_fn: Callable[[ExperimentConfig, str, int], str],
    train_solver_fn: Callable[[ExperimentConfig, str, int, str], str],
    commit_fn: Callable[[], None] = lambda: None,
    reload_fn: Optional[Callable[[], None]] = None,
) -> tuple[str, str]:
    """Alternate challenger and solver phases for `cfg.run.num_iters` rounds.

    `train_challenger_fn(cfg, run_ts, iteration)` and
    `train_solver_fn(cfg, run_ts, iteration, challenger_ckpt)` run one phase
    each and return the merged checkpoint path. In `modal_app.py` they
    dispatch to remote GPU functions; in tests they're fakes. The `cfg` they
    receive already has `challenger.model_path`/`solver.model_path` swapped
    to the current iteration's checkpoints.

    Preemption-safe exactly like the old Modal orchestrator: state is saved
    to `paths.state_file()` (same JSON schema) and `commit_fn` (e.g. Modal
    volume.commit) is called after every phase; on retry the loop resumes
    from the recorded phase. Returns (final_challenger, final_solver).
    """
    base_challenger = cfg.challenger.model_path
    base_solver = cfg.solver.model_path
    state_file = RunPaths(storage, cfg.run.abbr, run_ts="_", iteration=1).state_file()

    state = CoevolveState.load(state_file)
    if state is not None:
        print(f"[RESUME] iter={state.next_iter} phase={state.next_phase} run_ts={state.run_ts}")
        print(f"[RESUME]   challenger : {state.current_challenger}")
        print(f"[RESUME]   solver     : {state.current_solver}")
    else:
        run_ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        state = CoevolveState(
            run_ts=run_ts,
            wandb_group=f"coevolve_{cfg.run.abbr}_{run_ts}",
            next_iter=1,
            next_phase="challenger",
            current_challenger=base_challenger,
            current_solver=base_solver,
        )
        print(f"[FRESH] Starting co-evolution run_ts={run_ts}")

    os.environ.setdefault("WANDB_RUN_GROUP", state.wandb_group)
    _wandb_summary_run(cfg, RunPaths(storage, cfg.run.abbr, state.run_ts, 1), state.wandb_group)

    print(
        f"=== Creative Co-Evolution | abbr={cfg.run.abbr} iters={cfg.run.num_iters} "
        f"challenger={state.current_challenger} solver={state.current_solver} "
        f"seed={cfg.run.seed} profile={cfg.run.profile} ==="
    )

    if cfg.judge.type == "sft-critic":
        # One warm-up for the whole run, not per phase/iteration — the
        # deployed service's scaledown_window is sized to comfortably span
        # the gap between phases (REFACTOR_PLAN.md §6.2).
        wait_for_sft_critic_judge(cfg.judge.critic_url)

    for iter_num in range(1, cfg.run.num_iters + 1):
        if iter_num < state.next_iter:
            print(f"[SKIP] Iteration {iter_num} already complete")
            continue

        paths = RunPaths(storage, cfg.run.abbr, state.run_ts, iter_num)
        print(f"\n{'=' * 58}")
        print(f" CO-EVOLUTION ITERATION  {iter_num} / {cfg.run.num_iters}  (run: {state.run_ts})")
        print(f"   challenger init : {state.current_challenger}")
        print(f"   solver init     : {state.current_solver}")
        print(f"{'=' * 58}")

        iter_cfg = deepcopy(cfg)
        iter_cfg.challenger.model_path = state.current_challenger
        iter_cfg.solver.model_path = state.current_solver

        # ---- Phase A: challenger (skipped when resuming at this iteration's solver phase) ----
        if iter_num == state.next_iter and state.next_phase == "solver":
            challenger_ckpt = state.current_challenger
            print(f"\n[RESUME] Phase A already done — using: {challenger_ckpt}")
        else:
            challenger_ckpt = train_challenger_fn(iter_cfg, state.run_ts, iter_num)
            verify_checkpoint(f"Challenger iter{iter_num}", challenger_ckpt, reload_fn)
            upload_parquets_to_hf(paths, "challenger")
            state.next_phase = "solver"
            state.current_challenger = challenger_ckpt
            state.save(state_file)
            commit_fn()
            print(f"[STATE] Committed after Phase A iter {iter_num}")

        # ---- Phase B: solver ----
        solver_ckpt = train_solver_fn(iter_cfg, state.run_ts, iter_num, challenger_ckpt)
        verify_checkpoint(f"Solver iter{iter_num}", solver_ckpt, reload_fn)
        upload_parquets_to_hf(paths, "solver")
        state.current_challenger = challenger_ckpt
        state.current_solver = solver_ckpt
        state.next_iter = iter_num + 1
        state.next_phase = "challenger"
        state.save(state_file)
        commit_fn()
        print(f"[STATE] Committed after Phase B iter {iter_num}")

        print(f"\n=== Iteration {iter_num} complete ===")
        print(f"    Challenger : {state.current_challenger}")
        print(f"    Solver     : {state.current_solver}")

    # Clean completion — remove state so the next run with this abbr starts fresh.
    if state_file.exists():
        state_file.unlink()
    commit_fn()

    print(f"\n{'=' * 58}")
    print(" Creative Co-Evolution Complete")
    print(f" Iterations       : {cfg.run.num_iters}")
    print(f" Final challenger : {state.current_challenger}")
    print(f" Final solver     : {state.current_solver}")
    print(f"{'=' * 58}")
    return state.current_challenger, state.current_solver
