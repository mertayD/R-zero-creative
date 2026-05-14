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
# the Modal secret injected as environment variables (HF_TOKEN, WANDB_API_KEY,
# ANTHROPIC_API_KEY). Add `"anthropic": "sk-ant-..."` to tokens.json so the
# WritingBench eval Function below can call Claude-Sonnet-4-5 as the judge.
# ---------------------------------------------------------------------------
if modal.is_local():
    with open("tokens.json") as _f:
        _t = json.load(_f)
    HF_TOKEN          = _t.get("huggingface", "")
    WANDB_API_KEY     = _t.get("wandb", "")
    ANTHROPIC_API_KEY = _t.get("anthropic", "")
else:
    # In container: values are already in env via the secret
    HF_TOKEN          = os.environ.get("HF_TOKEN", "")
    WANDB_API_KEY     = os.environ.get("WANDB_API_KEY", "")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Infrastructure config from .env
# ---------------------------------------------------------------------------
APP_NAME            = os.getenv("APP_NAME",              "r-zero-main")
VOLUME_NAME         = os.getenv("VOLUME_NAME",           "r-zero-storage")
REMOTE_REPO_PATH    = os.getenv("REMOTE_REPO_PATH",      "/root/R-Zero")
REMOTE_STORAGE_PATH = os.getenv("REMOTE_STORAGE_PATH",   "/storage")
MODAL_FULL_GPU      = os.getenv("MODAL_FULL_GPU",        "A100-40GB:8")
# Eval is a single 4B model in vLLM + Claude API calls — one A100 is plenty.
MODAL_EVAL_GPU      = os.getenv("MODAL_EVAL_GPU",        "A100-40GB:1")
MODAL_EVAL_TIMEOUT  = int(os.getenv("MODAL_EVAL_TIMEOUT_SECONDS", "10800"))  # 3h
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
    "WANDB_API_KEY":     WANDB_API_KEY,
    "HF_TOKEN":          HF_TOKEN,          # also available via tokens.json mount
    "HUGGINGFACENAME":   HUGGINGFACENAME,
    "STORAGE_PATH":      REMOTE_STORAGE_PATH,
    # Used only by eval_writing_bench (Claude-Sonnet-4-5 judge). Empty in
    # tokens.json is OK — only fails when eval is actually invoked.
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
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
        # vLLM+Inductor autotune cache under /tmp can JSONDecodeError across runs; smoke uses eager anyway.
        "TORCHINDUCTOR_MAX_AUTOTUNE":       "0",
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
    smoke_questions_per_gpu: int = 8,
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
    smoke_questions_per_gpu: int = 8,
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


# ---------------------------------------------------------------------------
# Eval-only Function: WritingBench (and any future writing/creative bench)
# ---------------------------------------------------------------------------
# Reuses the same image, volume, and secret as run_smoke_test — the only
# differences are the GPU count (1 vs 8) and the timeout (3 h default vs 24 h).
#
# Two ways to call this:
#
#   1) From the CLI (one-shot evaluation of any model checkpoint):
#        modal run --detach modal_run.py::eval --model Qwen/Qwen3-4B-Base \
#            --subset smoke50
#
#   2) Programmatically from another Modal Function (e.g., between R-Zero
#      training iterations) — same Modal app, so .remote() works directly:
#        from modal_run import eval_writing_bench
#        eval_writing_bench.remote(
#            model=f"{REMOTE_STORAGE_PATH}/models/qwen3-4b-iter1/.../huggingface",
#            subset="mid100",
#        )
# ---------------------------------------------------------------------------
@app.function(
    gpu=MODAL_EVAL_GPU,
    image=image,
    volumes={REMOTE_STORAGE_PATH: volume},
    secrets=[runtime_secret],
    timeout=MODAL_EVAL_TIMEOUT,
    env={
        # Keep the same vLLM hygiene flags used by training to avoid stale
        # autotune cache crashes across runs.
        "VLLM_DISABLE_COMPILE_CACHE":      "1",
        "TORCHINDUCTOR_MAX_AUTOTUNE":       "0",
        "TOKENIZERS_PARALLELISM":           "true",
        "VLLM_LOGGING_LEVEL":               "WARN",
        "PYTHONUNBUFFERED":                 "1",
    },
)
def eval_writing_bench(
    model: str = "Qwen/Qwen3-4B-Base",
    subset: str = "smoke50",
):
    """
    Run scripts/eval_writing_bench.sh against a model checkpoint on Modal.

    Args:
        model:  HF id (e.g. "Qwen/Qwen3-4B-Base") or absolute path on the
                Modal volume to a merged checkpoint
                (e.g. "/storage/models/qwen3-4b-iter1/.../huggingface").
        subset: "smoke50" | "mid100" | "full".

    Outputs land at ${STORAGE_PATH}/writing_bench/<model_slug>/<subset>/ on
    the persistent volume. Pull them after the run with:
        modal volume get r-zero-storage writing_bench/<model_slug>/<subset>/scores.xlsx
    """
    import os
    import subprocess

    repo    = REMOTE_REPO_PATH
    storage = os.environ["STORAGE_PATH"]

    # Sanity-check the judge key surfaces a clear error before we burn GPU time.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is empty. Add an 'anthropic' entry to "
            "tokens.json locally and re-run; modal_run.py will inject it "
            "via the runtime secret."
        )

    os.chdir(repo)

    # Cache HF downloads on the persistent volume — same trick training uses.
    hf_cache = f"{storage}/hf_cache"
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"]              = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache

    env = {
        **os.environ,
        "STORAGE_PATH": storage,
        "HF_HOME": hf_cache,
        "HUGGINGFACE_HUB_CACHE": hf_cache,
    }

    print(f"=== WritingBench eval | model={model} subset={subset} ===")
    print(f"    storage = {storage}")
    print(f"    outputs -> {storage}/writing_bench/<model_slug>/{subset}/")

    # Always commit, even on failure: stage 1 (generation) is the expensive
    # part, and we want its responses.jsonl to survive a stage-2 (judge) crash
    # so the rerun can skip generation. Modal also flushes the volume on
    # container exit, but this is the explicit guarantee.
    try:
        result = subprocess.run(
            ["bash", "scripts/eval_writing_bench.sh", model, subset, storage],
            env=env,
            cwd=repo,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"WritingBench eval pipeline exited with code {result.returncode}"
            )
        print("=== WritingBench eval complete ===")
    finally:
        volume.commit()
        print("    artefacts committed to volume")


@app.local_entrypoint()
def eval(
    model: str = "Qwen/Qwen3-4B-Base",
    subset: str = "smoke50",
):
    """
    Local entrypoint: `modal run --detach modal_run.py::eval --model X --subset Y`.

    Just forwards to eval_writing_bench.remote(...). Detach mode is recommended
    for `full` and `mid100` — judge calls take a while and you don't want a
    dropped terminal to kill the run.
    """
    eval_writing_bench.remote(model=model, subset=subset)


# ---------------------------------------------------------------------------
# vLLM-based ModelClient for Prompt Generation
# ---------------------------------------------------------------------------
# Implements the ModelClient interface using vLLM for local inference.
# Used by ChallengerPromptPipeline for actual model-based generation.
# ---------------------------------------------------------------------------

class VLLMClient:
    """
    vLLM-based model client for prompt generation.

    Uses vLLM to load a model once and reuse it across multiple generation calls.
    Follows the pattern from question_generate.py.
    """

    def __init__(self, model_name: str, seed: int = 42, gpu_memory_utilization: float = 0.8):
        """
        Initialize vLLM client.

        Args:
            model_name: HuggingFace model ID (e.g., "Qwen/Qwen3-4B-Base")
            seed: Random seed for reproducibility
            gpu_memory_utilization: GPU memory fraction to use (0-1)
        """
        import vllm
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = vllm.LLM(
            model=model_name,
            tokenizer=model_name,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def generate(self, prompts: list, max_tokens: int = 4096) -> list:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of prompt strings
            max_tokens: Maximum tokens per response

        Returns:
            List of generated responses

        Raises:
            GenerationError subclasses on failures
        """
        import vllm

        try:
            sampling_params = vllm.SamplingParams(
                max_tokens=max_tokens,
                temperature=1.0,
                top_p=0.95,
                n=1,
                stop_token_ids=[self.tokenizer.eos_token_id],
            )

            completions = self.model.generate(prompts, sampling_params=sampling_params)
            return [completion.outputs[0].text for completion in completions]

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            from question_generate import NetworkError
            raise NetworkError(f"vLLM generation failed: {e}")
        except Exception as e:
            from question_generate import NetworkError
            raise NetworkError(f"Unexpected error during vLLM generation: {e}")

    def format_prompt(self, messages: list) -> str:
        """
        Format chat messages into prompt string.

        Args:
            messages: List of dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True,
            )
        else:
            # Fallback for models without chat template
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])


# ---------------------------------------------------------------------------
# Challenger Prompt Generation Function: WritingBench Creative Prompts
# ---------------------------------------------------------------------------
# Generates challenger prompts for creative writing tasks using WritingBench
# framework with vLLM model-based inference for actual generation.
#
# Usage:
#   modal run --detach modal_run.py::generate_challenger_prompts --num-prompts 5
#
# ---------------------------------------------------------------------------
@app.function(
    gpu=MODAL_EVAL_GPU,
    image=image,
    volumes={REMOTE_STORAGE_PATH: volume},
    secrets=[runtime_secret],
    timeout=MODAL_EVAL_TIMEOUT,
    env={
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
        "TOKENIZERS_PARALLELISM": "true",
        "VLLM_LOGGING_LEVEL": "WARN",
        "PYTHONUNBUFFERED": "1",
    },
)
def generate_challenger_prompts(
    num_prompts: int = 1,
    model_name: str = "Qwen/Qwen3-4B-Base",
    seed: int = 42,
    output_dir: str = "generated_question",
    gpu_memory_utilization: float = 0.8,
):
    """
    Generate challenger prompts using WritingBench framework with vLLM inference.

    Workflow:
        1. Initialize vLLM with the specified model
        2. Create ChallengerPromptPipeline with vLLM client
        3. Generate prompts with actual model-based inference
        4. Save results to persistent volume with generation logs

    Args:
        num_prompts: Number of prompts to generate
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen3-4B-Base")
        seed: Random seed for reproducibility (shared across batch)
        output_dir: Directory name on volume for outputs (relative to STORAGE_PATH)
        gpu_memory_utilization: GPU memory fraction (0-1)

    Returns:
        Dict with batch metadata, generation stats, and error summary
    """
    import os
    import sys
    import json
    from pathlib import Path

    repo = REMOTE_REPO_PATH
    storage = os.environ["STORAGE_PATH"]
    output_path = f"{storage}/{output_dir}"

    os.chdir(repo)
    sys.path.insert(0, repo)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    print(f"=== Challenger Prompt Generation with vLLM ===")
    print(f"  model: {model_name}")
    print(f"  num_prompts: {num_prompts}")
    print(f"  seed: {seed}")
    print(f"  output_dir: {output_path}")

    try:
        # Import pipeline
        from question_generate import ChallengerPromptPipeline

        # Step 1: Initialize vLLM client
        print(f"\n[Step 1] Initializing vLLM with {model_name}...")
        model_client = VLLMClient(
            model_name=model_name,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print(f"✓ vLLM initialized")

        # Step 2: Create pipeline with vLLM
        print(f"\n[Step 2] Creating ChallengerPromptPipeline...")
        pipeline = ChallengerPromptPipeline(
            model_client=model_client,
            language="English",
            num_criteria=5,
            seed=seed,
            max_retries=3,
            retry_wait=1.0,
        )
        print(f"✓ Pipeline created")

        # Step 3: Generate batch with actual model inference
        print(f"\n[Step 3] Generating {num_prompts} prompt(s) with model inference...")
        batch = pipeline.generate_batch(num_prompts=num_prompts)

        # Summary stats
        log = batch.generation_log
        print(f"\n✓ Generation complete:")
        print(f"  Successfully generated: {log['total_generated']}/{log['total_attempted']}")
        print(f"  Skipped: {log['skipped']}")
        print(f"  JSON parse failures: {log['json_parse_failures']}")
        print(f"  Network failures: {log['network_failures']}")

        if batch.domains_sampled:
            print(f"\n  Domains sampled: {', '.join(batch.domains_sampled)}")
        if batch.subdomains_sampled:
            subdomains_display = ', '.join(batch.subdomains_sampled[:5])
            if len(batch.subdomains_sampled) > 5:
                subdomains_display += f", ... ({len(batch.subdomains_sampled)} total)"
            print(f"  Subdomains sampled: {subdomains_display}")

        # Step 4: Save batch to volume
        output_file = Path(output_path) / f"{batch.batch_id}.json"
        with open(output_file, 'w') as f:
            f.write(batch.to_json())
        print(f"\n✓ Batch saved to {output_file}")

        # Display sample prompt
        if batch.prompts:
            sample = batch.prompts[0]
            print(f"\n[Sample Prompt]")
            print(f"  ID: {sample.prompt_id}")
            print(f"  Domain: {sample.domain_name} / {sample.subdomain}")
            print(f"  Initial Query (first 100 chars): {sample.initial_query[:100]}...")
            print(f"  Refined Query (first 100 chars): {sample.refined_query[:100]}...")
            print(f"  Guidance Applied: {sample.guidance_applied}")
            print(f"  Criteria Generated: {bool(sample.evaluation_criteria)}")

        return {
            "batch_id": batch.batch_id,
            "total_attempted": log["total_attempted"],
            "total_generated": log["total_generated"],
            "skipped": log["skipped"],
            "json_parse_failures": log["json_parse_failures"],
            "network_failures": log["network_failures"],
            "output_file": str(output_file),
            "domains": batch.domains_sampled,
            "num_subdomains": len(batch.subdomains_sampled),
        }

    except Exception as e:
        print(f"✗ Error during generation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Commit changes to persistent volume
        volume.commit()
        print(f"\n✓ Results committed to volume")


@app.local_entrypoint()
def generate_prompts(
    num_prompts: int = 1,
    model_name: str = "Qwen/Qwen3-4B-Base",
    seed: int = 42,
    output_dir: str = "generated_question",
):
    """
    Local entrypoint: `modal run modal_run.py::generate_prompts --num-prompts 5 --model-name Qwen/Qwen3-4B-Base`.

    Generates challenger prompts using WritingBench framework with vLLM inference.

    Uses .remote() (not .spawn()) because prompt generation is part of the R-Zero
    training pipeline's Challenger phase, which requires synchronous execution and
    passing results to downstream components (Solver, Judge).
    """
    result = generate_challenger_prompts.remote(
        num_prompts=num_prompts,
        model_name=model_name,
        seed=seed,
        output_dir=output_dir,
    )
    print(f"\n=== Generation Complete ===")
    print(json.dumps(result, indent=2))
