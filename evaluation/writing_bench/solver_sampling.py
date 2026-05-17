"""
Solver Sampling Pipeline

Orchestrates M samples per prompt using vLLM generation + Claude/Critic evaluation.

Pipeline:
  1. Load prompts from JSON (from one_shot_creative_question_generate.py output)
  2. For each prompt: Generate M responses using vLLM
  3. Evaluate all M responses using WritingBench criteria
  4. Aggregate scores and compute R-Zero uncertainty reward
  5. Output WritingPromptResponse[] to JSONL + rewards summary

Usage:
    python evaluation/writing_bench/solver_sampling.py \
        --model Qwen/Qwen3-4B \
        --input_json creative_generated_samples/one_shot.json \
        --output_dir evaluation/writing_bench/solver_sampling_outputs/run_20250515_m5 \
        --M 5 \
        --evaluator claude
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from tqdm import tqdm

# Setup paths
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import vllm
from transformers import AutoTokenizer

# Import shared data structures
from evaluation.shared import WritingPromptResponse, SampleResult

# Import existing evaluation code
from evaluation.writing_bench.evaluate_benchmark import EvalAgent
from evaluation.writing_bench.evaluator import ClaudeAgent, CriticAgent
from evaluation.writing_bench.prompt import evaluate_system

# Import WritingPrompt class
from question_generate.one_shot_creative_question_generate import WritingPrompt


def load_prompts_from_json(json_path: str) -> List[WritingPrompt]:
    """
    Load prompts from one_shot.json (output of one_shot_creative_question_generate.py).
    Deserializes to WritingPrompt objects using from_dict().

    Expected schema:
    {
        "batch_id": "...",
        "prompts": [
            {
                "prompt_id": "prompt_0001",
                "domain": "D6",
                "domain_name": "Advertising & Marketing",
                "subdomain": "Product Description",
                "query": "Craft a compelling...",
                "criteria": [...],
                "format_score": 1
            }
        ]
    }

    Returns:
        List[WritingPrompt]: Deserialized prompt objects
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    prompts_data = data.get("prompts", [])
    prompts = [WritingPrompt.from_dict(p) for p in prompts_data]
    print(f"[solver_sampling] Loaded {len(prompts)} prompts from {json_path}")
    return prompts


def initialize_vllm(
    model_id: str,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
) -> tuple:
    """
    Initialize vLLM model and tokenizer (reusing pattern from generate_responses_vllm.py).

    Args:
        model_id: HuggingFace model ID
        gpu_memory_utilization: GPU memory utilization ratio
        tensor_parallel_size: Tensor parallelism size
        max_model_len: Override vLLM max_model_len if needed

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"[initialize_vllm] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[initialize_vllm] Loading vLLM model: {model_id}")
    llm_kwargs = dict(
        model=model_id,
        tokenizer=model_id,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    model = vllm.LLM(**llm_kwargs)
    return model, tokenizer


def build_prompt_for_generation(tokenizer, query: str) -> str:
    """
    Build prompt for vLLM generation (same as generate_responses_vllm.py).

    WritingBench provides full instruction in query; no system prompt added.
    """
    chat = [{"role": "user", "content": query}]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
    return query


def generate_samples(
    model,
    tokenizer,
    query: str,
    M: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    max_tokens: int = 16000,
) -> List[str]:
    """
    Generate M response samples for a single query.

    Args:
        model: vLLM LLM instance
        tokenizer: HF tokenizer
        query: Writing task query
        M: Number of samples
        temperature, top_p, top_k, max_tokens: Sampling parameters (from WritingBench)

    Returns:
        List of M response strings
    """
    prompt = build_prompt_for_generation(tokenizer, query)

    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id]
        if tokenizer.eos_token_id is not None
        else None,
    )

    # Generate M samples in a single batch call
    outputs = model.generate([prompt] * M, sampling_params=sampling_params)

    responses = [out.outputs[0].text for out in outputs]
    return responses


def evaluate_sample(
    eval_agent: EvalAgent, response_text: str, query: str, criterion: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a single sample against one criterion using Claude/Critic.

    Returns:
        {"score": int, "reason": str}
    """
    try:
        score_result = eval_agent.generate_score(
            {"response": response_text}, query, criterion
        )
        return score_result
    except Exception as e:
        print(f"[evaluate_sample] Error: {e}")
        return {"score": 0, "reason": f"Evaluation failed: {e}"}


def process_prompt(
    prompt: WritingPrompt,
    model,
    tokenizer,
    eval_agent: EvalAgent,
    M: int = 5,
) -> WritingPromptResponse:
    """
    Process one prompt: generate M samples and evaluate all.

    Args:
        prompt: WritingPrompt object from one_shot.json
        model: vLLM LLM
        tokenizer: HF tokenizer
        eval_agent: Evaluation agent (Claude or Critic)
        M: Number of samples

    Returns:
        WritingPromptResponse with all M samples + scores

    FORMAT CHECK (R-Zero paper):
    If format_score == -1 (question fails format validation):
    - Skip all sampling and evaluation
    - Return WritingPromptResponse with M=0, samples=[], reward overall=0.0
    """
    # FORMAT CHECK: Skip sampling if question failed format validation
    if prompt.format_score == -1:
        print(
            f"  [{prompt.prompt_id}] ⚠️ Format check failed (format_score={prompt.format_score}). "
            f"Skipping sampling. Reward = 0.0"
        )
        # Create response with no samples but with format_score set
        writing_response = WritingPromptResponse(
            prompt_id=prompt.prompt_id,
            domain=prompt.domain,
            domain_name=prompt.domain_name,
            subdomain=prompt.subdomain,
            query=prompt.query,
            criteria=prompt.criteria,
            format_score=prompt.format_score,
            samples=[],
            M=0,
        )
        # Directly set reward score without calling compute_reward_uncertainty()
        # (avoid unnecessary computation for invalid formats)
        writing_response.reward_score = {
            "overall": 0.0,
            "format": -1.0,
            "accuracy": 0.0,
        }
        return writing_response

    # Step 1: Generate M samples
    print(f"  [{prompt.prompt_id}] Generating {M} samples...")
    responses = generate_samples(model, tokenizer, prompt.query, M=M)

    # Step 2: Evaluate all M samples
    print(f"  [{prompt.prompt_id}] Evaluating {M} samples × {len(prompt.criteria)} criteria...")

    samples: List[SampleResult] = []
    for sample_idx, response_text in enumerate(responses):
        sample_scores: Dict[str, Dict[str, Any]] = {}

        for criterion in prompt.criteria:
            criterion_name = criterion.get("name", f"Criterion {len(sample_scores)}")
            score_result = evaluate_sample(
                eval_agent, response_text, prompt.query, criterion
            )
            sample_scores[criterion_name] = score_result

        sample = SampleResult(sample_id=sample_idx, text=response_text, scores=sample_scores)
        samples.append(sample)

    # Step 3: Create WritingPromptResponse
    writing_response = WritingPromptResponse(
        prompt_id=prompt.prompt_id,
        domain=prompt.domain,
        domain_name=prompt.domain_name,
        subdomain=prompt.subdomain,
        query=prompt.query,
        criteria=prompt.criteria,
        format_score=prompt.format_score,
        samples=samples,
        M=M,
    )

    # Step 4: Compute statistics and reward
    writing_response.compute_statistics()
    reward_score = writing_response.compute_reward_uncertainty()

    print(
        f"  [{prompt.prompt_id}] ✓ Score: {writing_response.compute_aggregate_score():.2f}, "
        f"Reward (overall): {reward_score['overall']:.3f}"
    )

    return writing_response


def save_responses(
    responses: List[WritingPromptResponse], output_jsonl_path: str
) -> None:
    """Save WritingPromptResponse[] to JSONL file."""
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, "w") as f:
        for response in responses:
            f.write(response.to_jsonl() + "\n")
    print(f"[save] Wrote {len(responses)} responses to {output_jsonl_path}")


def compute_rewards_summary(
    responses: List[WritingPromptResponse],
) -> Dict[str, Any]:
    """
    Compute summary statistics over all rewards.

    Returns:
        {
            "num_prompts": int,
            "M": int,
            "reward_mean": float,
            "reward_std": float,
            "reward_min": float,
            "reward_max": float,
            "score_mean": float,
            "score_std": float,
            "top_5_by_reward": [...],
            "bottom_5_by_reward": [...]
        }
    """
    if not responses:
        return {}

    # Extract overall rewards (R-Zero uncertainty)
    overall_rewards = [
        r.reward_score["overall"] for r in responses if r.reward_score is not None
    ]
    scores = [r.compute_aggregate_score() for r in responses]

    from statistics import mean, stdev

    M = responses[0].M if responses else 0

    top_5 = sorted(
        responses,
        key=lambda r: r.reward_score["overall"] if r.reward_score else 0,
        reverse=True
    )[:5]
    bottom_5 = sorted(
        responses,
        key=lambda r: r.reward_score["overall"] if r.reward_score else 0
    )[:5]

    return {
        "num_prompts": len(responses),
        "M": M,
        "reward_mean": mean(overall_rewards) if overall_rewards else 0.0,
        "reward_std": stdev(overall_rewards) if len(overall_rewards) > 1 else 0.0,
        "reward_min": min(overall_rewards) if overall_rewards else 0.0,
        "reward_max": max(overall_rewards) if overall_rewards else 0.0,
        "score_mean": mean(scores) if scores else 0.0,
        "score_std": stdev(scores) if len(scores) > 1 else 0.0,
        "score_min": min(scores) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
        "top_5_by_reward": [
            {
                "prompt_id": r.prompt_id,
                "reward_overall": r.reward_score["overall"],
                "reward_accuracy": r.reward_score["accuracy"],
                "score": r.compute_aggregate_score(),
            }
            for r in top_5
        ],
        "bottom_5_by_reward": [
            {
                "prompt_id": r.prompt_id,
                "reward_overall": r.reward_score["overall"],
                "reward_accuracy": r.reward_score["accuracy"],
                "score": r.compute_aggregate_score(),
            }
            for r in bottom_5
        ],
    }


def save_rewards_summary(
    responses: List[WritingPromptResponse], output_json_path: str
) -> None:
    """Save rewards summary to JSON."""
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    summary = compute_rewards_summary(responses)
    with open(output_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] Wrote rewards summary to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Solver sampling: generate M samples per prompt and evaluate."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HF model ID or path for generation.",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Input JSON file (from one_shot_creative_question_generate.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=5,
        help="Number of samples per prompt.",
    )
    parser.add_argument(
        "--evaluator",
        choices=["claude", "critic"],
        default="claude",
        help="Evaluation agent.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16000,
        help="Max tokens for generation.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Solver Sampling Pipeline")
    print("=" * 80)

    # Step 1: Load prompts
    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    prompts = load_prompts_from_json(args.input_json)

    # Step 2: Initialize vLLM model and tokenizer
    model, tokenizer = initialize_vllm(
        model_id=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Step 3: Setup evaluation agent
    print(f"[setup] Setting up {args.evaluator} evaluator")
    if args.evaluator == "claude":
        agent = ClaudeAgent(system_prompt=evaluate_system)
    else:
        agent = CriticAgent(system_prompt=evaluate_system)
    eval_agent = EvalAgent(agent)

    # Step 4: Process all prompts
    print(f"\n[processing] Processing {len(prompts)} prompts with M={args.M}")
    responses: List[WritingPromptResponse] = []

    for prompt in tqdm(prompts, desc="Solver sampling"):
        try:
            response = process_prompt(
                prompt,
                model,
                tokenizer,
                eval_agent,
                M=args.M,
            )
            responses.append(response)
        except Exception as e:
            print(f"[error] Failed to process {prompt.prompt_id}: {e}")
            import traceback
            traceback.print_exc()

    # Step 5: Save results
    print(f"\n[output] Saving {len(responses)} responses")
    os.makedirs(args.output_dir, exist_ok=True)

    responses_jsonl = os.path.join(args.output_dir, "responses.jsonl")
    rewards_json = os.path.join(args.output_dir, "rewards_summary.json")

    save_responses(responses, responses_jsonl)
    save_rewards_summary(responses, rewards_json)

    # Step 6: Print summary
    print(f"\n{'=' * 80}")
    print("Solver Sampling Complete")
    print(f"{'=' * 80}")
    print(f"Prompts processed: {len(responses)}")
    print(f"Samples per prompt (M): {args.M}")
    print(f"Total evaluations: {len(responses) * args.M}")
    print(f"\nReward Statistics:")
    if responses and responses[0].reward_score is not None:
        overall_rewards = [r.reward_score["overall"] for r in responses]
        from statistics import mean, stdev
        print(f"  Mean overall reward: {mean(overall_rewards):.3f}")
        print(f"  Std overall reward: {stdev(overall_rewards):.3f}")
        print(f"  Range: [{min(overall_rewards):.3f}, {max(overall_rewards):.3f}]")

    print(f"\nOutputs:")
    print(f"  Responses: {responses_jsonl}")
    print(f"  Summary: {rewards_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
