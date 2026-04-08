"""
Modal deployment for R-Zero pipeline on columbia-daplab.

Usage (smoke test — end-to-end validation):

    # Always use --detach for long GPU jobs. Without it, closing the terminal or
    # losing the network stops the Modal app and kills training mid-flight.
    modal run --detach modal_run.py

    # Custom model / smoke scale (all smoke-* args optional; defaults are small):
    modal run --detach modal_run.py --base-model Qwen/Qwen3-4B-Base --abbr qwen3-4b-smoke

    # Equivalent wrapper from repo root:
    bash scripts/modal_run_smoke_detach.sh

Tokens are read from tokens.json (huggingface + wandb) and infrastructure
config (volume name, GPU type, etc.) is read from .env.
"""

import json
import os
import modal
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Tokens — read locally from tokens.json; in the container they come from
# the Modal secret injected as environment variables (HF_TOKEN, WANDB_API_KEY).
# ---------------------------------------------------------------------------
if modal.is_local():
    with open("tokens.json") as _f:
        _t = json.load(_f)
    HF_TOKEN      = _t.get("huggingface", "")
    WANDB_API_KEY = _t.get("wandb", "")
else:
    # In container: values are already in env via the secret
    HF_TOKEN      = os.environ.get("HF_TOKEN", "")
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

# ---------------------------------------------------------------------------
# Infrastructure config from .env
# ---------------------------------------------------------------------------
APP_NAME            = os.getenv("APP_NAME",              "r-zero-main")
VOLUME_NAME         = os.getenv("VOLUME_NAME",           "r-zero-storage")
REMOTE_REPO_PATH    = os.getenv("REMOTE_REPO_PATH",      "/root/R-Zero")
REMOTE_STORAGE_PATH = os.getenv("REMOTE_STORAGE_PATH",   "/storage")
MODAL_FULL_GPU      = os.getenv("MODAL_FULL_GPU",        "A100-40GB:8")
MODAL_TIMEOUT       = int(os.getenv("MODAL_TIMEOUT_SECONDS", "86400"))
HUGGINGFACENAME     = os.getenv("HUGGINGFACENAME",       "")

# ---------------------------------------------------------------------------
# Persistent volume (created on first run if absent)
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Container image
# Layered so Modal can cache each step independently.
# ---------------------------------------------------------------------------
image = (
    # CUDA 12.6 devel – provides nvcc + all CUDA libs for compilation
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install([
        "git", "wget", "curl",
        "build-essential", "ninja-build",
        "libaio-dev", "pkg-config",
    ])
    # PyTorch from the official CUDA-specific wheel index
    .pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        "torchaudio==2.7.0",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    # vLLM pulls in most deep-learning deps automatically
    .pip_install("vllm==0.9.1")
    # flash-attn: install wheel first so source builds work, then try flash-attn
    .run_commands(
        "pip install wheel && "
        "(pip install flash-attn==2.7.4.post1 --no-build-isolation "
        "|| pip install flash-attn --no-build-isolation "
        "|| echo '[WARN] flash-attn install failed — sdpa will be used')"
    )
    # Project-specific packages not bundled by vLLM
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
    # Mount the local repo into the image at container-start time (copy=False
    # means files are injected on startup, not baked into the image layer).
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
        ),
    )
)

# ---------------------------------------------------------------------------
# Runtime secrets – injected as env vars inside the container
# ---------------------------------------------------------------------------
runtime_secret = modal.Secret.from_dict({
    "WANDB_API_KEY":   WANDB_API_KEY,
    "HF_TOKEN":        HF_TOKEN,          # also available via tokens.json mount
    "HUGGINGFACENAME": HUGGINGFACENAME,
    "STORAGE_PATH":    REMOTE_STORAGE_PATH,
})

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME)


@app.function(
    gpu=MODAL_FULL_GPU,
    image=image,
    volumes={REMOTE_STORAGE_PATH: volume},
    secrets=[runtime_secret],
    timeout=MODAL_TIMEOUT,
    # Forward environment variables used by sub-scripts
    env={
        "VLLM_DISABLE_COMPILE_CACHE":      "1",
        "TOKENIZERS_PARALLELISM":           "true",
        "NCCL_DEBUG":                       "WARN",
        "VLLM_LOGGING_LEVEL":               "WARN",
        "TORCH_NCCL_AVOID_RECORD_STREAMS":  "1",
        "PYTORCH_CUDA_ALLOC_CONF":          "expandable_segments:False",
        "PYTHONUNBUFFERED":                 "1",
        # Ensure local vLLM reward callbacks never go through a broken HTTP proxy.
        "NO_PROXY":                         "127.0.0.1,localhost,0.0.0.0",
        "no_proxy":                         "127.0.0.1,localhost,0.0.0.0",
    },
)
def run_smoke_test(
    base_model: str = "Qwen/Qwen3-4B-Base",
    abbr: str = "qwen3-4b-smoke",
    smoke_max_train_samples: int = 8,
    smoke_max_val_samples: int = 8,
    # Multiples of 8: questioner uses 4 GPUs, solver 8 — verl requires even DataProto chunking.
    smoke_rollout_batch_size: int = 8,
    smoke_val_batch_size: int = 8,
    smoke_global_batch_size: int = 8,
    smoke_questioner_max_steps: int = 2,
    smoke_solver_max_steps: int = 4,
    smoke_questions_per_gpu: int = 3,
):
    """
    Run one full R-Zero iteration (questioner + solver) on a small data budget
    (default 8 training/val rows, short optimization) to validate the pipeline.
    """
    import os
    import subprocess
    import sys

    # ---- Paths & env ----
    repo      = REMOTE_REPO_PATH
    storage   = os.environ["STORAGE_PATH"]
    hf_name   = os.environ.get("HUGGINGFACENAME", "")
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    os.chdir(repo)
    sys.path.insert(0, repo)

    # ---- Create storage subdirectories on the volume ----
    for d in ["evaluation", "models", "generated_question", "temp_results"]:
        os.makedirs(f"{storage}/{d}", exist_ok=True)

    # tokens.json is already present in the repo mount with real values —
    # no need to overwrite it here.

    # Cache HuggingFace model downloads on the persistent volume so they
    # survive across Modal runs and don't need to be re-fetched each time.
    hf_cache = f"{storage}/hf_cache"
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"]             = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache

    # ---- Propagate env vars so child bash scripts pick them up ----
    env = {
        **os.environ,
        "STORAGE_PATH": storage,
        "HUGGINGFACENAME": hf_name,
        "WANDB_API_KEY": wandb_key,
        "HF_HOME": hf_cache,
        "HUGGINGFACE_HUB_CACHE": hf_cache,
        "SMOKE_MAX_TRAIN_SAMPLES": str(smoke_max_train_samples),
        "SMOKE_MAX_VAL_SAMPLES": str(smoke_max_val_samples),
        "SMOKE_ROLLOUT_BATCH_SIZE": str(smoke_rollout_batch_size),
        "SMOKE_VAL_BATCH_SIZE": str(smoke_val_batch_size),
        "SMOKE_GLOBAL_BATCH_SIZE": str(smoke_global_batch_size),
        "SMOKE_QUESTIONER_MAX_STEPS": str(smoke_questioner_max_steps),
        "SMOKE_SOLVER_MAX_STEPS": str(smoke_solver_max_steps),
        "SMOKE_QUESTIONS_PER_GPU": str(smoke_questions_per_gpu),
    }

    print(
        f"=== R-Zero SMOKE TEST | model={base_model} abbr={abbr} | "
        f"train_cap={smoke_max_train_samples} q_steps={smoke_questioner_max_steps} "
        f"s_steps={smoke_solver_max_steps} rollout_bs={smoke_rollout_batch_size} ==="
    )
    result = subprocess.run(
        ["bash", "scripts/smoke_test.sh", base_model, abbr],
        env=env,
        cwd=repo,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Smoke test pipeline exited with code {result.returncode}"
        )

    # Persist any new artefacts written to the volume
    volume.commit()
    print("=== Smoke test complete — artefacts committed to volume ===")


# ---------------------------------------------------------------------------
# Local entry-point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3-4B-Base",
    abbr: str = "qwen3-4b-smoke",
    smoke_max_train_samples: int = 8,
    smoke_max_val_samples: int = 8,
    # Multiples of 8: questioner uses 4 GPUs, solver 8 — verl requires even DataProto chunking.
    smoke_rollout_batch_size: int = 8,
    smoke_val_batch_size: int = 8,
    smoke_global_batch_size: int = 8,
    smoke_questioner_max_steps: int = 2,
    smoke_solver_max_steps: int = 4,
    smoke_questions_per_gpu: int = 3,
):
    run_smoke_test.remote(
        base_model=base_model,
        abbr=abbr,
        smoke_max_train_samples=smoke_max_train_samples,
        smoke_max_val_samples=smoke_max_val_samples,
        smoke_rollout_batch_size=smoke_rollout_batch_size,
        smoke_val_batch_size=smoke_val_batch_size,
        smoke_global_batch_size=smoke_global_batch_size,
        smoke_questioner_max_steps=smoke_questioner_max_steps,
        smoke_solver_max_steps=smoke_solver_max_steps,
        smoke_questions_per_gpu=smoke_questions_per_gpu,
    )
