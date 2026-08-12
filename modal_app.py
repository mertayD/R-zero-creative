"""modal_app.py — thin Modal wrappers around the creative_rzero package (T2.8).

All pipeline logic lives in `creative_rzero/orchestrator.py`; this file only
defines Modal infrastructure (image/volume/secrets), the shared GPU-function
environment, and three functions that load a typed config and delegate:

    train_challenger  — one Phase A (challenger) training pass, one GPU box
    train_solver      — one Phase B (solver) training pass, one GPU box
    coevolve          — CPU loop alternating the two via .remote, resumable

Usage:

    # Always --detach for GPU jobs; the entrypoint spawns and exits.
    modal run --detach modal_app.py::main \
        --config configs/exp/validation_tiny.yaml \
        --set "run.abbr=my-run,challenger.max_steps=2"

GPU sizing: the GPU count must be >= the config's
`verl.trainer.n_gpus_per_node + 1` (training GPUs + one aux GPU for the
vLLM solver oracle / prompt generation — see orchestrator.py). Set
CREATIVE_GPU in .env or the shell, e.g. CREATIVE_GPU="A100-40GB:2" for
configs with n_gpus_per_node=1.
"""

import os
import sys

import modal
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Infrastructure config — shares the volume (checkpoints, HF cache, logs)
# with the legacy modal_run.py app, but registers under its own app name so
# deploys never clobber each other during the migration.
# ---------------------------------------------------------------------------
APP_NAME            = os.getenv("CREATIVE_APP_NAME",    "r-zero-creative")
VOLUME_NAME         = os.getenv("VOLUME_NAME",          "r-zero-storage")
REMOTE_REPO_PATH    = os.getenv("REMOTE_REPO_PATH",     "/root/R-Zero")
REMOTE_STORAGE_PATH = os.getenv("REMOTE_STORAGE_PATH",  "/storage")
GPU_SPEC            = os.getenv("CREATIVE_GPU",         "A100-40GB:2")

# In the container this file lives at /root/modal_app.py while the repo is
# mounted at REMOTE_REPO_PATH — make the repo importable at *module import*
# time (argument deserialization runs before any function body gets to call
# _container_setup, so anything imported during unpickling needs this).
if os.path.isdir(REMOTE_REPO_PATH) and REMOTE_REPO_PATH not in sys.path:
    sys.path.insert(0, REMOTE_REPO_PATH)
TIMEOUT_S           = int(os.getenv("CREATIVE_TIMEOUT_SECONDS", os.getenv("MODAL_TIMEOUT_SECONDS", "86400")))
RETRIES             = int(os.getenv("CREATIVE_RETRIES", "0"))

HF_TOKEN            = os.environ.get("HF_TOKEN", "")
WANDB_API_KEY       = os.environ.get("WANDB_API_KEY", "")
PERPLEXITY_API_KEY  = os.environ.get("PERPLEXITY_API_KEY", "")
HUGGINGFACENAME     = os.getenv("HUGGINGFACENAME", "")

# sft-critic judge service (judge.type=sft-critic, REFACTOR_PLAN.md §6) —
# deployed independently of any training run, see critic_judge_server below.
# No auth: only we deploy this, and it's only ever up for the span of a
# training run (min_containers=0 + scaledown_window below scale it back to
# zero shortly after), not a long-lived public service.
CRITIC_MODEL_ID              = "AQuarterMile/WritingBench-Critic-Model-Qwen-7B"
CRITIC_SERVED_NAME           = os.getenv("CRITIC_SERVED_NAME", "writingbench-critic-qwen-7b")
CRITIC_JUDGE_GPU             = os.getenv("CRITIC_JUDGE_GPU", "L4")
CRITIC_JUDGE_SCALEDOWN_WINDOW_S = int(os.getenv("CRITIC_JUDGE_SCALEDOWN_WINDOW_S", "900"))

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Same image as modal_run.py (kept in lockstep until that file is deleted
# post-T2.9); layered so Modal caches each step independently.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install([
        "git", "wget", "curl",
        "build-essential", "ninja-build",
        "libaio-dev", "pkg-config",
    ])
    .pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        "torchaudio==2.7.0",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("vllm==0.9.1")
    .run_commands(
        "pip install wheel && "
        "(pip install flash-attn==2.7.4.post1 --no-build-isolation "
        "|| pip install flash-attn --no-build-isolation "
        "|| echo '[WARN] flash-attn install failed — sdpa will be used')"
    )
    .pip_install([
        "ray==2.46.0",
        "accelerate==1.7.0",
        "datasets==3.6.0",
        "wandb==0.20.1",
        "mathruler==0.1.0",
        "math-verify==0.7.0",
        "stopit",
        "scikit-learn==1.7.0",
        "nltk==3.9.1",
        "einops==0.8.1",
        "peft==0.15.2",
        "omegaconf==2.3.0",
        "huggingface-hub==0.32.4",
        "flask==3.1.1",
        "regex==2024.11.6",
        "matplotlib==3.10.3",
        "pandas==2.3.0",
        "transformers==4.52.4",
        "tokenizers==0.21.1",
        "liger_kernel==0.5.10",
        "tensordict==0.8.3",
        "python-dotenv==1.1.0",
        "requests==2.32.4",
        "qwen-vl-utils==0.0.11",
        "pylatexenc==2.10",
        "tabulate==0.9.0",
        "jsonlines==4.0.0",
        "latex2sympy2_extended==1.10.1",
        "torchdata==0.11.0",
        "codetiming",
    ])
    .add_local_dir(
        ".",
        remote_path=REMOTE_REPO_PATH,
        copy=False,
        ignore=modal.FilePatternMatcher(
            ".git/**",
            "**/__pycache__/**",
            "**/*.pyc",
            ".env",
            "*.egg-info/**",
            ".venv/**",
        ),
    )
)

runtime_secret = modal.Secret.from_dict({
    "WANDB_API_KEY":      WANDB_API_KEY,
    "HF_TOKEN":           HF_TOKEN,
    "HUGGINGFACENAME":    HUGGINGFACENAME,
    "STORAGE_PATH":       REMOTE_STORAGE_PATH,
    "PERPLEXITY_API_KEY": PERPLEXITY_API_KEY,
})

app = modal.App(APP_NAME)

# The one shared environment for GPU functions (was duplicated across three
# env dicts in modal_run.py).
GPU_RUNTIME_ENV = {
    "VLLM_DISABLE_COMPILE_CACHE":      "1",
    "TORCHINDUCTOR_MAX_AUTOTUNE":      "0",
    "TOKENIZERS_PARALLELISM":          "true",
    "NCCL_DEBUG":                      "WARN",
    "VLLM_LOGGING_LEVEL":              "WARN",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "PYTORCH_CUDA_ALLOC_CONF":         "expandable_segments:False",
    "PYTHONUNBUFFERED":                "1",
    "NO_PROXY":                        "127.0.0.1,localhost,0.0.0.0",
    "no_proxy":                        "127.0.0.1,localhost,0.0.0.0",
}


def gpu_function(f):
    """The decorator factory for phase-training GPU functions — one place
    for GPU spec, image, volume, secrets, timeout, and runtime env."""
    return app.function(
        gpu=GPU_SPEC,
        image=image,
        volumes={REMOTE_STORAGE_PATH: volume},
        secrets=[runtime_secret],
        timeout=TIMEOUT_S,
        retries=RETRIES,
        env=GPU_RUNTIME_ENV,
    )(f)


def _container_setup(wandb_group: str = "") -> str:
    """Shared per-container init: repo on sys.path + cwd, HF cache on the
    volume, W&B mode. Returns the storage root."""
    import sys

    os.chdir(REMOTE_REPO_PATH)
    sys.path.insert(0, REMOTE_REPO_PATH)
    os.environ["REMOTE_REPO_PATH"] = REMOTE_REPO_PATH

    storage = os.environ["STORAGE_PATH"]
    hf_cache = f"{storage}/hf_cache"
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache

    if wandb_group and os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "online"
        os.environ["WANDB_RUN_GROUP"] = wandb_group
    else:
        os.environ.setdefault("WANDB_MODE", "disabled")
    return storage


# Only primitives (str/int/list) cross .remote() boundaries: Modal pickles
# arguments by reference, and the receiving container deserializes them
# BEFORE the function body can set up sys.path — passing an ExperimentConfig
# instance died there with "No module named 'creative_rzero'". Each GPU
# function loads the typed config itself from the repo mounted in its own
# container ("functions become cfg = load(...)" — REFACTOR_PLAN T2.8).


@gpu_function
def train_challenger(config: str, overrides: list[str] = [], run_ts: str = "",
                     iteration: int = 1, wandb_group: str = "") -> str:
    """One Phase A pass: challenger.model_path trains against the frozen
    solver.model_path oracle. Returns the merged checkpoint path."""
    storage = _container_setup(wandb_group)
    from creative_rzero import orchestrator
    from creative_rzero.config import load
    from creative_rzero.paths import RunPaths

    cfg = load(config, cli_args=list(overrides))
    paths = RunPaths(storage, cfg.run.abbr, run_ts, iteration)
    ckpt = orchestrator.run_challenger_phase(cfg, paths)
    volume.commit()
    print(f"=== Challenger training complete — checkpoint: {ckpt} ===")
    return ckpt


@gpu_function
def train_solver(config: str, overrides: list[str] = [], run_ts: str = "",
                 iteration: int = 1, challenger_ckpt: str = "", wandb_group: str = "") -> str:
    """One Phase B pass: solver.model_path trains on prompts generated by
    challenger_ckpt. Returns the merged checkpoint path."""
    if not challenger_ckpt:
        raise ValueError("challenger_ckpt must be set to a merged challenger checkpoint path or HF model id")
    storage = _container_setup(wandb_group)
    from creative_rzero import orchestrator
    from creative_rzero.config import load
    from creative_rzero.paths import RunPaths

    cfg = load(config, cli_args=list(overrides))
    paths = RunPaths(storage, cfg.run.abbr, run_ts, iteration)
    ckpt = orchestrator.run_solver_phase(cfg, paths, challenger_ckpt)
    volume.commit()
    print(f"=== Solver training complete — checkpoint: {ckpt} ===")
    return ckpt


@app.function(
    image=image,  # reuses the training image (already has vllm==0.9.1 + the
                  # repo mounted) rather than a second, separately-maintained
                  # build target — heavier than strictly necessary for a
                  # single served model, deliberately kept simple for now.
    gpu=CRITIC_JUDGE_GPU,
    min_containers=0,               # scale to zero between runs — no GPU billed while idle
    scaledown_window=CRITIC_JUDGE_SCALEDOWN_WINDOW_S,
    timeout=86400,
)
@modal.web_server(port=8000, startup_timeout=600)
def critic_judge_server():
    """Persistent vLLM OpenAI-compatible server for the sft-critic judge
    (judge.type=sft-critic — REFACTOR_PLAN.md §6). Deploy once, independent
    of any training run:

        modal deploy modal_app.py

    The resulting URL (`https://<workspace>--<app>-critic-judge-server.modal.run`)
    is stable for the life of this deployment regardless of whether a
    container is currently running behind it — set it once as
    `judge.critic_url` in the experiment config. `min_containers=0` means
    Modal's own autoscaling gives start-before-training/stop-after behavior
    for free: no container (no GPU cost) until the first request of a run,
    `scaledown_window` idle seconds after the last one before it scales back
    to zero. `orchestrator.py::wait_for_sft_critic_judge` warms this up once
    per co-evolution run, before Phase A, so the cold start (if any) doesn't
    stall the first judge call mid-training.

    No auth on this endpoint: only we deploy it, and it's only ever up for
    the span of a training run (min_containers=0/scaledown_window above), not
    a long-lived public service. If that changes, add vLLM's --api-key.
    """
    import subprocess

    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", CRITIC_MODEL_ID,
        "--served-model-name", CRITIC_SERVED_NAME,
        "--port", "8000",
        "--dtype", "auto",
        "--disable-log-requests",
    ])


@app.function(
    image=image,
    volumes={REMOTE_STORAGE_PATH: volume},
    secrets=[runtime_secret],
    timeout=TIMEOUT_S,
    retries=RETRIES,
    env={"PYTHONUNBUFFERED": "1"},
)
def coevolve(config: str, overrides: list[str] = []) -> list[str]:
    """CPU orchestrator: loads the typed config and runs the co-evolution
    loop, dispatching each phase to a GPU function. Resumable across Modal
    retries via the state file on the volume."""
    storage = _container_setup()
    from creative_rzero import orchestrator
    from creative_rzero.config import load

    cfg = load(config, cli_args=list(overrides))

    def _phase_overrides(iter_cfg) -> list[str]:
        """Serialize the per-iteration state (current checkpoint paths) back
        into dotlist overrides so only primitives cross the .remote() boundary;
        the GPU function re-loads the same config from them."""
        return list(overrides) + [
            f"challenger.model_path={iter_cfg.challenger.model_path}",
            f"solver.model_path={iter_cfg.solver.model_path}",
        ]

    final = orchestrator.run_coevolve(
        cfg,
        storage,
        train_challenger_fn=lambda c, ts, i: train_challenger.remote(
            config, _phase_overrides(c), ts, i,
            wandb_group=os.environ.get("WANDB_RUN_GROUP", ""),
        ),
        train_solver_fn=lambda c, ts, i, ckpt: train_solver.remote(
            config, _phase_overrides(c), ts, i, challenger_ckpt=ckpt,
            wandb_group=os.environ.get("WANDB_RUN_GROUP", ""),
        ),
        commit_fn=volume.commit,
        reload_fn=volume.reload,
    )
    return list(final)


@app.function(
    image=image,
    volumes={REMOTE_STORAGE_PATH: volume},
    secrets=[runtime_secret],
    timeout=1800,
    env={"PYTHONUNBUFFERED": "1"},
)
def preflight(config: str, overrides: list[str] = []) -> str:
    """CPU-only wiring check — everything that can fail *before* GPU spend:

      1. typed config load + validation
      2. materialized verl YAML for both roles merges cleanly into verl's own
         PPOConfig schema (an unknown key here would crash an hour into a run)
      3. the reward entrypoint imports the real caller chain the way verl
         loads it, with the config-to-env bridge applied
      4. the judge client constructs (catches a missing PERPLEXITY_API_KEY)
    """
    import importlib.util

    storage = _container_setup()
    results = []

    from creative_rzero.config import load, save_resolved
    from creative_rzero.paths import RunPaths
    from creative_rzero.steps.train_verl import materialize_verl_config

    cfg = load(config, cli_args=list(overrides))
    results.append(f"[1] config load ok (abbr={cfg.run.abbr}, profile={cfg.run.profile})")

    tmp = "/tmp/preflight"
    paths = RunPaths(storage, cfg.run.abbr, "preflight", 1)

    from omegaconf import OmegaConf
    from verl.trainer.config import PPOConfig

    for role in ("challenger", "solver"):
        yaml_path = materialize_verl_config(cfg, paths, role, tmp)
        merged = OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.load(yaml_path))
        ppo = OmegaConf.to_object(merged)
        ppo.deep_post_init()
        assert ppo.worker.reward.reward_function is not None, "reward_function nulled by post_init"
        assert ppo.worker.reward.reward_function_kwargs == {"role": role}
        results.append(f"[2] verl PPOConfig schema ok ({role}): {yaml_path}")

    resolved = save_resolved(cfg, tmp)
    os.environ["EXPERIMENT_CONFIG_PATH"] = str(resolved)
    entry_path = f"{REMOTE_REPO_PATH}/creative_rzero/rewards/verl_entry.py"
    spec = importlib.util.spec_from_file_location("custom_reward_fn", entry_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # exactly how verl's FunctionRewardManager loads it
    for role in ("challenger", "solver"):
        impl = module._get_impl(role)
        results.append(f"[3] reward impl import ok ({role}): {impl.__module__}")

    from evaluation.writing_bench.evaluator.llm import ClaudeAgent

    agent = ClaudeAgent("preflight")
    results.append(f"[4] judge client ok (model={agent.model})")

    report = "\n".join(results)
    print(report)
    return report


@app.local_entrypoint()
def preflight_check(config: str = "configs/exp/validation_tiny.yaml", set: str = ""):  # noqa: A002
    """Run the CPU wiring check: modal run modal_app.py::preflight_check"""
    overrides = [s.strip() for s in set.split(",") if s.strip()]
    print(preflight.remote(config, overrides))


@app.local_entrypoint()
def main(config: str = "configs/exp/validation_tiny.yaml", set: str = ""):  # noqa: A002 — Modal CLI arg name
    """Validate the config locally (fail in seconds, not after a container
    spin-up), then spawn the co-evolution loop.

    `--set` is a comma-separated dotlist, e.g.
    `--set "run.abbr=my-run,challenger.max_steps=2"`.
    """
    overrides = [s.strip() for s in set.split(",") if s.strip()]
    try:
        from creative_rzero.config import load

        cfg = load(config, cli_args=overrides)
        n_gpus = int(cfg.verl["trainer"]["n_gpus_per_node"])
        gpu_count = int(GPU_SPEC.rsplit(":", 1)[1]) if ":" in GPU_SPEC else 1
        if gpu_count < n_gpus + 1:
            raise SystemExit(
                f"CREATIVE_GPU={GPU_SPEC!r} provides {gpu_count} GPUs but config needs "
                f"verl.trainer.n_gpus_per_node+1 = {n_gpus + 1} (training + aux vLLM GPU)"
            )
        print(f"[OK] config valid: abbr={cfg.run.abbr} profile={cfg.run.profile} "
              f"iters={cfg.run.num_iters} gpus={GPU_SPEC}")
    except ImportError as e:
        print(f"[WARN] skipping local config validation (missing dep: {e}); the container will validate")

    fc = coevolve.spawn(config, overrides)
    print(f"=== Co-evolution submitted (function call: {fc.object_id}) ===")
    print(f"    modal app logs {APP_NAME}   # to follow")
