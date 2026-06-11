"""
One-Shot Challenger Prompt Generation
======================================

Simplified single-prompt approach to WritingBench challenger prompt generation.

Instead of three-step pipeline (initial query → refinement → criteria),
all logic is embedded in the system prompt. The model generates:
  - Private reasoning (thinking about refinements)
  - Output query (polished, detailed writing prompt)
  - Evaluation criteria (structured assessment framework)

This maintains WritingBench quality while keeping integration simple for R-Zero.

Uses the WritingBench framework:
  - WRITING_DOMAINS: D1-D6 domain types with subdomains
  - QUERY_REFINEMENT_GUIDANCE_POOL: Refinement principles applied contextually
  - Format validation: Strict <output>...</output> + JSON format
"""

import argparse
import json
import os
import sys
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import random

# Setup paths
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import regex as re
import vllm
from transformers import AutoTokenizer

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _wandb_active() -> bool:
    return _WANDB_AVAILABLE and _wandb.run is not None

# Import WritingBench framework
from question_generate.creative_writing_prompts import (
    WRITING_DOMAINS,
    QUERY_REFINEMENT_GUIDANCE_POOL,
)

STORAGE_PATH = os.getenv("STORAGE_PATH")

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Error Classes (match creative_question_generate.py)
# =============================================================================

class GenerationError(Exception):
    """Base generation error."""
    pass


class JSONParseError(GenerationError):
    """JSON output is malformed → RETRY."""
    pass


class NetworkError(GenerationError):
    """vLLM/transient issue → RETRY."""
    pass


class SkippableError(GenerationError):
    """Unrecoverable → SKIP."""
    pass


# =============================================================================
# Format Validation (match creative_question_generate.py)
# =============================================================================

class FormatValidator:
    """Validates model output format and provides scoring."""

    @staticmethod
    def validate_output_tags(text: str) -> tuple[bool, str]:
        """Check if output has proper <output>...</output> tags."""
        match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
        if match:
            return True, match.group(1).strip()
        return False, ""

    @staticmethod
    def validate_json(text: str) -> tuple[bool, dict | list | None]:
        """Check if text is valid JSON."""
        try:
            parsed = json.loads(text)
            return True, parsed
        except (json.JSONDecodeError, ValueError):
            return False, None

    @staticmethod
    def validate_response(response: str) -> tuple[int, dict | list | None]:
        """
        Validate complete response format (XML tags + JSON).

        Returns:
            Tuple of (score, parsed_json)
            score: 1 if valid, -1 if invalid format
        """
        has_tags, extracted = FormatValidator.validate_output_tags(response)
        if not has_tags:
            logger.warning(f"Missing <output> tags in response")
            return -1, None

        is_valid_json, parsed = FormatValidator.validate_json(extracted)
        if not is_valid_json:
            logger.warning(f"Invalid JSON in output block: {extracted[:200]}")
            return -1, None

        # Ensure the top-level value is a dict with a string query field.
        # The model can occasionally emit `"query": {...}` instead of a plain
        # string, which would cause callers to crash on .strip().
        if not isinstance(parsed, dict):
            logger.warning(f"Parsed JSON is not a dict: {type(parsed)}")
            return -1, None
        query_val = parsed.get("query")
        if query_val is not None and not isinstance(query_val, str):
            logger.warning(
                f"'query' field is {type(query_val).__name__}, expected str; "
                f"value preview: {str(query_val)[:120]}"
            )
            return -1, None

        return 1, parsed


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class QueryRequirements:
    """Represents style, format, and length requirements."""
    style: Optional[str] = None
    format: Optional[str] = None
    length: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


@dataclass
class WritingPrompt:
    """One-shot generated writing prompt with metadata."""

    prompt_id: str
    domain: str  # D1-D6
    domain_name: str
    subdomain: str
    query: str  # The actual writing prompt
    criteria: List[Dict[str, Any]] = field(default_factory=list)
    requirements: QueryRequirements = field(default_factory=QueryRequirements)

    # Metadata
    guidance_applied: List[str] = field(default_factory=list)
    format_score: int = 1
    language: str = "English"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['requirements'] = self.requirements.to_dict()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WritingPrompt":
        """Deserialize WritingPrompt from dictionary."""
        req_data = data.get("requirements", {})
        requirements = QueryRequirements(
            style=req_data.get("style"),
            format=req_data.get("format"),
            length=req_data.get("length"),
        )

        return cls(
            prompt_id=data.get("prompt_id", ""),
            domain=data.get("domain", ""),
            domain_name=data.get("domain_name", ""),
            subdomain=data.get("subdomain", ""),
            query=data["query"],
            criteria=data.get("criteria", []),
            requirements=requirements,
            guidance_applied=data.get("guidance_applied", []),
            format_score=data.get("format_score", 1),
            language=data.get("language", "English"),
            seed=data.get("seed", 42),
        )


@dataclass
class PromptBatch:
    """Collection of prompts generated in a batch."""

    batch_id: str
    prompts: List[WritingPrompt] = field(default_factory=list)
    domains_sampled: List[str] = field(default_factory=list)
    subdomains_sampled: List[str] = field(default_factory=list)
    generation_log: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.generation_log:
            self.generation_log = {
                "total_attempted": 0,
                "total_generated": 0,
                "skipped": 0,
                "json_parse_failures": 0,
                "network_failures": 0,
                "format_validation_failures": 0,
            }

    def add_prompt(self, prompt: Optional[WritingPrompt], errors: Optional[List[Dict[str, Any]]] = None):
        """Add a prompt to the batch."""
        if prompt is not None:
            self.prompts.append(prompt)
            self.generation_log["total_generated"] += 1
            if prompt.domain not in self.domains_sampled:
                self.domains_sampled.append(prompt.domain)
            if prompt.subdomain not in self.subdomains_sampled:
                self.subdomains_sampled.append(prompt.subdomain)
        else:
            self.generation_log["skipped"] += 1

        self.generation_log["total_attempted"] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'batch_id': self.batch_id,
            'domains_sampled': self.domains_sampled,
            'subdomains_sampled': self.subdomains_sampled,
            'generation_log': self.generation_log,
            'prompts': [p.to_dict() for p in self.prompts]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


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

    # Scoring rules for criteria (from WritingBench)
    scoring_rules = {
        "1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
        "3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
        "5-6": "Average score description: Adequate but not exemplary. Baseline performance that meets essential requirements. Most models may achieve this score.",
        "7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
        "9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."
    }

    system_prompt = f"""You are an expert writing task generator with deep knowledge of diverse writing domains and subdomains.

CRITICAL FORMATTING RULES:
1. You MUST wrap your entire response in <output> and </output> tags
2. Inside the tags, return ONLY valid JSON (no markdown, no code blocks, no explanations)
3. Do NOT include any text before the <output> tag
4. Do NOT include any text after the </output> tag
5. Do NOT use unescaped quotes inside JSON strings
6. Your response MUST be parseable as valid JSON

Do NOT violate these rules under any circumstances."""

    user_prompt = f"""Generate ONE detailed writing prompt for the subdomain "{subdomain}" within {domain_name}.

DOMAIN CONTEXT:
- Domain: {domain_name}
- Description: {domain_description}
- Subdomain: {subdomain}
- Language: {language}

INTERNAL REASONING STAGE (private, do not output):

STEP 1: IDEATION & CONTEXT
- Think about the {domain_name} domain and specifically the "{subdomain}" subdomain
- Consider realistic, detailed, and specific writing requests appropriate for this context
- Ensure the requests reflect the domain's standards and typical use cases

STEP 2: APPLY REFINEMENT PRINCIPLES
These refinement principles help enhance the prompt quality:
{guidance_text}

As you design the prompt, incorporate these principles to make it more specific, constrained, and valuable.

STEP 3: DESIGN EVALUATION CRITERIA
Create 5 strict evaluation criteria that can distinguish subtle differences in response quality.

Each criterion MUST include:
- name: A concise criterion name
- criteria_description: Detailed description emphasizing what the criterion evaluates
- "1-2": Critical deficiencies and major issues
- "3-4": Below average - noticeable shortcomings
- "5-6": Average - adequate but not exemplary
- "7-8": Above average - competent execution
- "9-10": High - exceptional performance

The criteria should emphasize:
- Relevance to the writing task
- Coherence and logical flow
- Depth and specificity
- Adherence to requirements
- Overall quality and effectiveness

STEP 4: IDENTIFY REQUIREMENTS
Look for and identify any:
- Style requirements (e.g., formal, casual, tone, audience)
- Format requirements (e.g., structure, template, outline)
- Length requirements (e.g., word count, page count, character limits)

OUTPUT STAGE - Return ONLY this JSON format (nothing else):
<output>
{{
  "query": "Your detailed, polished, specific writing prompt that reflects the refinement principles",
  "criteria": [
    {{
      "name": "Criterion 1 Name",
      "criteria_description": "Detailed description for the first criteria, emphasizing detailed and critical assessment.",
      "1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
      "3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
      "5-6": "Average score description: Adequate but not exemplary. Baseline performance that meets essential requirements.",
      "7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed.",
      "9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness."
    }},
    {{
      "name": "Criterion 2 Name",
      "criteria_description": "Detailed description for the second criteria...",
      "1-2": "...",
      "3-4": "...",
      "5-6": "...",
      "7-8": "...",
      "9-10": "..."
    }},
    {{
      "name": "Criterion 3 Name",
      "criteria_description": "...",
      "1-2": "...",
      "3-4": "...",
      "5-6": "...",
      "7-8": "...",
      "9-10": "..."
    }},
    {{
      "name": "Criterion 4 Name",
      "criteria_description": "...",
      "1-2": "...",
      "3-4": "...",
      "5-6": "...",
      "7-8": "...",
      "9-10": "..."
    }},
    {{
      "name": "Criterion 5 Name",
      "criteria_description": "...",
      "1-2": "...",
      "3-4": "...",
      "5-6": "...",
      "7-8": "...",
      "9-10": "..."
    }}
  ],
  "requirements": {{
    "style": "Style requirement if explicitly mentioned, null otherwise",
    "format": "Format requirement if explicitly mentioned, null otherwise",
    "length": "Length requirement if explicitly mentioned, null otherwise"
  }}
}}
</output>

CRITICAL REMINDERS:
- Do NOT output your internal reasoning. Only output the JSON block.
- Ensure the query is specific and detailed, reflecting the refinement principles you applied.
- Each 5 criterion must have all 5 score levels (1-2, 3-4, 5-6, 7-8, 9-10) with detailed descriptions.
- Be strict in criteria design to distinguish subtle differences in quality.
- Reference exact aspects of what makes responses succeed or fail at each level.
- Make sure outputs are in {language} language."""

    return system_prompt, user_prompt, applied_guidance


def validate_one_shot_response(response: str) -> tuple[bool, Optional[dict]]:
    """
    Validate one-shot response format with full WritingBench criteria structure.

    Returns:
        Tuple of (is_valid, parsed_json)
    """
    score, parsed = FormatValidator.validate_response(response)
    if score != 1:
        return False, None

    # Validate structure
    if not isinstance(parsed, dict):
        logger.warning("Response is not a dict")
        return False, None

    if "query" not in parsed or "criteria" not in parsed:
        logger.warning("Missing required fields: query or criteria")
        return False, None

    # Validate criteria structure
    criteria = parsed.get("criteria", [])
    if not isinstance(criteria, list) or len(criteria) == 0:
        logger.warning("Criteria is not a non-empty list")
        return False, None

    # Validate each criterion has required fields
    required_score_levels = ["1-2", "3-4", "5-6", "7-8", "9-10"]
    for criterion in criteria:
        if not isinstance(criterion, dict):
            logger.warning("Criterion is not a dict")
            return False, None

        if "name" not in criterion or "criteria_description" not in criterion:
            logger.warning("Criterion missing name or criteria_description")
            return False, None

        # Check for all score levels
        for level in required_score_levels:
            if level not in criterion:
                logger.warning(f"Criterion missing score level: {level}")
                return False, None

    return True, parsed


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
    sampler = DomainSampler(seed=seed)
    batch_id = f"one_shot_batch_{seed}_{int(time.time())}"
    batch = PromptBatch(batch_id=batch_id)

    prompt_counter = 0

    for sample_idx in range(num_prompts):
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
            )
        else:
            prompt = f"system: {system_prompt}\n\nuser: {user_prompt}"

        # Generate with retries
        attempt = 0
        format_valid = False
        response = None
        parsed_json = None

        while attempt < num_format_retries and not format_valid:
            sampling_params = vllm.SamplingParams(
                max_tokens=4096,
                temperature=1.0,
                top_p=0.95,
                n=1,
                stop_token_ids=[tokenizer.eos_token_id],
            )

            completions = model.generate([prompt], sampling_params=sampling_params)
            response = completions[0].outputs[0].text

            # Validate format (with full WritingBench criteria structure)
            is_valid, parsed_json = validate_one_shot_response(response)

            if is_valid:
                format_valid = True
                prompt_counter += 1
                prompt_id = f"prompt_{prompt_counter:04d}"

                # Extract requirements (from generated output)
                requirements_data = parsed_json.get("requirements", {})
                requirements = QueryRequirements(
                    style=requirements_data.get("style"),
                    format=requirements_data.get("format"),
                    length=requirements_data.get("length"),
                )

                # Create WritingPrompt with all metadata
                writing_prompt = WritingPrompt(
                    prompt_id=prompt_id,
                    domain=domain_key,
                    domain_name=domain_name,
                    subdomain=subdomain,
                    query=parsed_json['query'],
                    criteria=parsed_json.get('criteria', []),  # Full criteria with 1-10 scoring
                    requirements=requirements,
                    guidance_applied=applied_guidance,  # Track which guidance was applied
                    format_score=1,
                    seed=seed,
                )

                batch.add_prompt(writing_prompt)
                print(f"  [{sample_idx + 1}/{num_prompts}] ✓ {domain_key}/{subdomain} | criteria: {len(writing_prompt.criteria)} | guidance: {len(applied_guidance)}")

            else:
                attempt += 1
                if attempt < num_format_retries:
                    print(f"  [{sample_idx + 1}/{num_prompts}] Format validation failed, retrying...")
                else:
                    batch.generation_log["format_validation_failures"] += 1
                    batch.add_prompt(None)
                    print(f"  [{sample_idx + 1}/{num_prompts}] ✗ Format validation failed after {num_format_retries} attempts")

    return batch


# =============================================================================
# Main
# =============================================================================

def main(args):
    print("=" * 80)
    print("One-Shot WritingBench Prompt Generation")
    print("=" * 80)

    # Check STORAGE_PATH early, before expensive operations
    if not STORAGE_PATH:
        print("ERROR: STORAGE_PATH is not set", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer for {args.model}: {e}", file=sys.stderr)
        sys.exit(1)

    # Load vLLM model
    try:
        model = vllm.LLM(
            model=args.model,
            tokenizer=args.model,
            seed=args.seed,  # ✅ args.seed is already an int from argparse
        )
    except Exception as e:
        print(f"ERROR: Failed to load vLLM model {args.model}: {e}", file=sys.stderr)
        sys.exit(1)

    _coevolve_iter = int(os.getenv("COEVOLVE_ITERATION", "0"))
    _wandb_parent_run_id = os.getenv("WANDB_RUN_ID")
    _wandb_owned = False  # whether we created the run (and should finish it)

    if _WANDB_AVAILABLE and os.getenv("WANDB_MODE") != "disabled":
        if _wandb_parent_run_id:
            # Attach to the coevolve parent run created by creative_coevolve_smoke.sh
            _wandb.init(
                project=os.getenv("WANDB_PROJECT", "r-zero"),
                id=_wandb_parent_run_id,
                resume="allow",
            )
        else:
            # Standalone invocation — create our own run
            _wandb.init(
                project=os.getenv("WANDB_PROJECT", "r-zero"),
                job_type="prompt-generation",
                name=f"prompt-gen_{args.save_name}_{args.suffix}_iter{_coevolve_iter}",
                config={
                    "model": args.model,
                    "num_samples": args.num_samples,
                    "seed": args.seed,
                    "format_retries": args.format_retries,
                    "coevolve_iteration": _coevolve_iter,
                },
            )
            _wandb_owned = True

    print(f"\nGenerating {args.num_samples} writing prompts")
    print(f"Model: {args.model}\n")

    # Generate batch
    batch = generate_prompts_batch(
        model=model,
        tokenizer=tokenizer,
        num_prompts=args.num_samples,
        num_format_retries=args.format_retries,
        seed=args.seed,
    )

    out_dir = os.path.join(STORAGE_PATH, "generated_questions_one_shot")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{args.save_name}_{args.suffix}.json")

    with open(out_path, "w") as f:
        f.write(batch.to_json())

    # Print summary with detailed criteria information
    log = batch.generation_log

    # Calculate criteria stats
    total_criteria = sum(len(p.criteria) for p in batch.prompts)
    avg_criteria = total_criteria / len(batch.prompts) if batch.prompts else 0
    total_guidance = sum(len(p.guidance_applied) for p in batch.prompts)
    avg_guidance = total_guidance / len(batch.prompts) if batch.prompts else 0

    print(f"\n{'=' * 80}")
    print(f"One-Shot WritingBench Generation Complete")
    print(f"{'=' * 80}")
    print(f"\nGeneration Stats:")
    print(f"  Total generated: {log['total_generated']}/{log['total_attempted']}")
    print(f"  Skipped: {log['skipped']}")
    print(f"  Format validation failures: {log['format_validation_failures']}")

    print(f"\nDomain Coverage:")
    print(f"  Domains sampled: {', '.join(batch.domains_sampled)}")
    print(f"  Unique subdomains: {len(batch.subdomains_sampled)}")

    print(f"\nCriteria Structure:")
    print(f"  Total criteria generated: {total_criteria}")
    print(f"  Average criteria per prompt: {avg_criteria:.1f}")
    print(f"  Each criterion includes: name, description, 1-2, 3-4, 5-6, 7-8, 9-10")

    print(f"\nRefinement Guidance:")
    print(f"  Total guidance applications: {total_guidance}")
    print(f"  Average guidance per prompt: {avg_guidance:.1f}")

    if batch.prompts:
        sample = batch.prompts[0]
        print(f"\nSample Prompt (prompt_{sample.prompt_id}):")
        print(f"  Domain: {sample.domain_name} ({sample.domain})")
        print(f"  Subdomain: {sample.subdomain}")
        print(f"  Query (first 120 chars): {sample.query[:120]}...")
        if sample.criteria and len(sample.criteria) > 0:
            print(f"  Criteria count: {len(sample.criteria)}")
            print(f"  First criterion: {sample.criteria[0].get('name', 'N/A')}")
        else:
            print(f"  Criteria count: 0 (no criteria generated)")
        if sample.guidance_applied:
            print(f"  Applied guidance ({len(sample.guidance_applied)}): {', '.join(sample.guidance_applied[:2])}...")
        print(f"  Requirements: style={sample.requirements.style}, format={sample.requirements.format}, length={sample.requirements.length}")

    print(f"\nOutput:")
    print(f"  File: {out_path}")
    if os.path.exists(out_path):
        file_size = os.path.getsize(out_path)
        print(f"  Size: {file_size} bytes")
    else:
        print(f"  ERROR: Output file not found at {out_path}")
    print(f"{'=' * 80}")

    if _wandb_active():
        metrics = {
            "prompt_gen/total_generated": log["total_generated"],
            "prompt_gen/total_attempted": log["total_attempted"],
            "prompt_gen/skipped": log["skipped"],
            "prompt_gen/format_validation_failures": log["format_validation_failures"],
            "prompt_gen/success_rate": log["total_generated"] / max(log["total_attempted"], 1),
            "prompt_gen/avg_criteria_per_prompt": avg_criteria,
            "prompt_gen/avg_guidance_per_prompt": avg_guidance,
            "prompt_gen/unique_domains": len(batch.domains_sampled),
            "prompt_gen/unique_subdomains": len(batch.subdomains_sampled),
        }
        if batch.prompts:
            table = _wandb.Table(columns=["prompt_id", "domain", "subdomain", "query_preview", "num_criteria", "num_guidance"])
            for p in batch.prompts:
                table.add_data(p.prompt_id, p.domain, p.subdomain, p.query[:150], len(p.criteria), len(p.guidance_applied))
            metrics["prompt_gen/prompts"] = table
        _wandb.log(metrics, step=_coevolve_iter)
        if _wandb_owned:
            _wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-shot WritingBench prompt generation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="Model to use")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of prompts to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--suffix", type=str, default="test", help="Suffix for output file")
    parser.add_argument("--save_name", type=str, default="one_shot_writingbench", help="Base name for output")
    parser.add_argument("--format_retries", type=int, default=3, help="Format validation retries")

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"[{args.suffix}] one_shot_creative_question_generate failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise
