"""
Challenger Prompt Generation Pipeline
=====================================

This module implements a pipeline for generating challenger prompts using the WritingBench framework.
The pipeline generates writing queries, refines them with guidance, and creates evaluation criteria.

The output uses dataclasses to maintain structured metadata without string keys.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import random
import json
from enum import Enum
import sys
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Import the creative writing prompts module
sys.path.insert(0, str(Path(__file__).parent))
from creative_writing_prompts import (
    WRITING_DOMAINS,
    QUERY_REFINEMENT_GUIDANCE_POOL,
    REQUIREMENT_CATEGORIES,
    generate_initial_query_prompt,
    generate_refinement_prompt,
    generate_criteria_prompt,
    extract_output_block,
    validate_and_extract_json,
)


# =============================================================================
# Error Classes
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
# Format Validation & Scoring
# =============================================================================

class FormatValidator:
    """Validates model output format and provides scoring."""

    @staticmethod
    def validate_output_tags(text: str) -> tuple[bool, str]:
        """
        Check if output has proper <output>...</output> tags.

        Returns:
            Tuple of (is_valid, extracted_content)
        """
        import re
        match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
        if match:
            return True, match.group(1).strip()
        return False, ""

    @staticmethod
    def validate_json(text: str) -> tuple[bool, dict | list | None]:
        """
        Check if text is valid JSON.

        Returns:
            Tuple of (is_valid, parsed_json)
        """
        import json
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
            parsed_json: Parsed content if valid, None otherwise
        """
        # Check for XML tags
        has_tags, extracted = FormatValidator.validate_output_tags(response)
        if not has_tags:
            logger.warning(f"Missing <output> tags in response")
            return -1, None

        # Check for valid JSON inside tags
        is_valid_json, parsed = FormatValidator.validate_json(extracted)
        if not is_valid_json:
            logger.warning(f"Invalid JSON in output block: {extracted[:200]}")
            return -1, None

        # Both valid
        return 1, parsed


# =============================================================================
# Model Client Interface
# =============================================================================

class ModelClient(ABC):
    """Abstract base class for model inference clients."""

    @abstractmethod
    def generate(self, prompts: List[str], max_tokens: int = 4096) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of prompt strings
            max_tokens: Maximum tokens per response

        Returns:
            List of generated responses

        Raises:
            NetworkError: On transient network issues
            JSONParseError: On JSON parsing failures
        """
        pass

    @abstractmethod
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into prompt string.

        Args:
            messages: List of dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        pass


# =============================================================================
# Enums for Type Safety
# =============================================================================

class DomainType(Enum):
    """Primary domain types."""
    ACADEMIC_ENGINEERING = "D1"
    FINANCE_BUSINESS = "D2"
    POLITICS_LAW = "D3"
    LITERATURE_ARTS = "D4"
    EDUCATION = "D5"
    ADVERTISING_MARKETING = "D6"


class RequirementType(Enum):
    """Writing requirement types."""
    STYLE = "R1"
    FORMAT = "R2"
    LENGTH = "R3"


# =============================================================================
# Dataclass Definitions
# =============================================================================

@dataclass
class QueryRequirements:
    """Represents style, format, and length requirements for a query."""
    style: Optional[str] = None
    format: Optional[str] = None
    length: Optional[str] = None

    def has_any_requirement(self) -> bool:
        """Check if any requirement is specified."""
        return any([self.style, self.format, self.length])

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class WritingPrompt:
    """Complete representation of a writing prompt with all metadata."""

    # Core identifiers
    prompt_id: str
    domain: DomainType
    domain_name: str
    subdomain: str

    # TEMPLATES (prompts sent to the model)
    initial_query_template: str
    refinement_template: str
    criteria_template: str

    # ACTUAL GENERATED OUTPUTS (what model returned)
    initial_query: str
    refined_query: str

    # Format validation scores (1=valid, -1=invalid)
    initial_query_format_score: int = 1
    refined_query_format_score: int = 1
    criteria_format_score: int = 1

    # Requirements
    requirements: QueryRequirements = field(default_factory=QueryRequirements)

    # Evaluation criteria
    evaluation_criteria: Optional[List[Dict[str, Any]]] = None
    num_criteria: int = 5

    # Metadata
    language: str = "English"
    guidance_applied: List[str] = field(default_factory=list)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding private fields."""
        data = asdict(self)
        # Convert Enum to string value
        data['domain'] = self.domain.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PromptBatch:
    """Collection of prompts generated in a single batch."""

    batch_id: str
    prompts: List[WritingPrompt] = field(default_factory=list)
    num_prompts: int = 0
    domains_sampled: List[str] = field(default_factory=list)
    subdomains_sampled: List[str] = field(default_factory=list)
    generation_log: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize generation log."""
        if not self.generation_log:
            self.generation_log = {
                "total_attempted": 0,
                "total_generated": 0,
                "json_parse_failures": 0,
                "network_failures": 0,
                "skipped": 0,
                "error_details": [],
            }

    def add_prompt(self, prompt: Optional[WritingPrompt], errors: Optional[List[Dict[str, Any]]] = None):
        """
        Add a prompt to the batch, tracking errors.

        Args:
            prompt: WritingPrompt instance or None if generation failed
            errors: List of error dicts from generation stage
        """
        if prompt is not None:
            self.prompts.append(prompt)
            self.generation_log["total_generated"] += 1
        else:
            self.generation_log["skipped"] += 1

        self.generation_log["total_attempted"] += 1
        self.num_prompts += 1

        # Track domain and subdomain only if prompt was successful
        if prompt is not None:
            if prompt.domain.value not in self.domains_sampled:
                self.domains_sampled.append(prompt.domain.value)
            if prompt.subdomain not in self.subdomains_sampled:
                self.subdomains_sampled.append(prompt.subdomain)

        # Log error details
        if errors:
            for error in errors:
                error_type = error.get("type", "unknown")
                if "json_parse" in error_type:
                    self.generation_log["json_parse_failures"] += 1
                elif "network" in error_type:
                    self.generation_log["network_failures"] += 1
            self.generation_log["error_details"].extend(errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_id': self.batch_id,
            'num_prompts': self.num_prompts,
            'domains_sampled': self.domains_sampled,
            'subdomains_sampled': self.subdomains_sampled,
            'generation_log': self.generation_log,
            'prompts': [p.to_dict() for p in self.prompts]
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Domain & Subdomain Sampling
# =============================================================================

class DomainSampler:
    """Handles uniform sampling of domains and subdomains."""

    def __init__(self, seed: Optional[int] = 42):
        """Initialize sampler with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
        self.domains = list(DomainType)

    def sample_domain_subdomain_pair(self) -> tuple[DomainType, str]:
        """
        Uniformly sample a domain and subdomain pair.

        Returns:
            Tuple of (DomainType, subdomain_name)
        """
        # Randomly select domain
        domain_enum = random.choice(self.domains)
        domain_key = domain_enum.value

        # Get subdomains for this domain
        domain_info = WRITING_DOMAINS[domain_key]
        subdomains = domain_info['subdomains']

        # Randomly select subdomain
        subdomain = random.choice(subdomains)

        return domain_enum, subdomain

    def sample_n_pairs(self, n: int) -> List[tuple[DomainType, str]]:
        """Sample n domain-subdomain pairs."""
        return [self.sample_domain_subdomain_pair() for _ in range(n)]


# =============================================================================
# Prompt Generation Pipeline
# =============================================================================

class ChallengerPromptPipeline:
    """
    Pipeline for generating challenger prompts with model-based inference.

    Workflow:
        1. Sample domain/subdomain uniformly
        2. Generate initial writing query template, send to model for actual query
        3. Apply query refinement template, send to model for refined query
        4. Generate evaluation criteria template, send to model for actual criteria
        5. Return as WritingPrompt dataclass with both templates and outputs

    The model_client is required (no fallback to templates-only mode).
    """

    def __init__(self,
                 model_client: ModelClient,
                 language: str = "English",
                 num_criteria: int = 5,
                 seed: int = 42,
                 max_retries: int = 3,
                 retry_wait: float = 1.0):
        """
        Initialize the pipeline.

        Args:
            model_client: ModelClient instance (required, not optional)
            language: Language for query generation (default: "English")
            num_criteria: Number of evaluation criteria to generate (default: 5)
            seed: Random seed for reproducibility and batch consistency
            max_retries: Number of retries for transient errors (default: 3)
            retry_wait: Base wait time between retries in seconds (default: 1.0)

        Raises:
            ValueError: If model_client is None
        """
        if model_client is None:
            raise ValueError("model_client is required; template-only mode not supported")

        self.model_client = model_client
        self.language = language
        self.num_criteria = num_criteria
        self.seed = seed
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.sampler = DomainSampler(seed=seed)
        self.prompt_counter = 0

    def _generate_initial_query_template(self,
                                         domain_enum: DomainType,
                                         subdomain: str,
                                         num_queries: int = 1) -> str:
        """
        Generate the initial query template (prompt to send to model).

        Args:
            domain_enum: Domain enum type
            subdomain: Subdomain name
            num_queries: Number of queries to generate

        Returns:
            Prompt template string
        """
        domain_key = domain_enum.value
        domain_name = WRITING_DOMAINS[domain_key]['name']

        query_prompt = generate_initial_query_prompt(
            num_queries=num_queries,
            subdomain=subdomain,
            primary_domain=domain_name,
            language=self.language
        )

        return query_prompt

    def _generate_initial_query_with_model(self,
                                          domain_enum: DomainType,
                                          subdomain: str) -> tuple[str, str, int]:
        """
        Generate actual initial query by sending template to model.
        Retries up to 5 times if format validation fails.

        Args:
            domain_enum: Domain enum type
            subdomain: Subdomain name

        Returns:
            Tuple of (template, actual_generated_query, format_score)
            format_score: 1 if valid format, -1 if invalid (after 5 retries)

        Raises:
            GenerationError: If generation fails after retries
        """
        template = self._generate_initial_query_template(domain_enum, subdomain)

        try:
            # Generate with format validation and retry logic
            responses, format_scores = self._generate_with_format_retry(
                [template], "initial_query", max_format_retries=5
            )

            if responses and responses[0]:
                response = responses[0]
                format_score = format_scores[0]

                # Extract first query from the response
                format_score, parsed_json = FormatValidator.validate_response(response)
                try:
                    if parsed_json and isinstance(parsed_json, list) and len(parsed_json) > 0:
                        actual_query = parsed_json[0] if isinstance(parsed_json[0], str) else str(parsed_json[0])
                    else:
                        actual_query = str(parsed_json) if parsed_json else response
                except (TypeError, AttributeError):
                    actual_query = response

                return template, actual_query, format_score
            else:
                raise SkippableError("Empty response from model")
        except GenerationError as e:
            logger.error(f"Initial query generation failed: {e}")
            raise

    def _generate_with_retry(self,
                            prompts: List[str],
                            stage_name: str,
                            max_tokens: int = 4096) -> List[str]:
        """
        Generate responses with retry logic for transient errors.

        Args:
            prompts: List of prompts to send to model
            stage_name: Name of generation stage (for logging)
            max_tokens: Max tokens per response

        Returns:
            List of generated responses (same length as prompts)

        Raises:
            GenerationError: If final attempt fails
        """
        error_log = {
            "stage": stage_name,
            "total_attempts": 0,
            "json_parse_errors": 0,
            "network_errors": 0,
        }

        for attempt in range(self.max_retries):
            try:
                error_log["total_attempts"] = attempt + 1
                responses = self.model_client.generate(prompts, max_tokens=max_tokens)
                return responses

            except JSONParseError as e:
                error_log["json_parse_errors"] += 1
                logger.warning(
                    f"[{stage_name}] JSON parse error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_wait * (attempt + 1))
                    continue
                else:
                    raise

            except NetworkError as e:
                error_log["network_errors"] += 1
                logger.warning(
                    f"[{stage_name}] Network error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_wait * (attempt + 1))
                    continue
                else:
                    raise

        raise SkippableError(f"Failed to generate {stage_name} after {self.max_retries} attempts")

    def _generate_with_format_retry(self,
                                   prompts: List[str],
                                   stage_name: str,
                                   max_tokens: int = 4096,
                                   max_format_retries: int = 5) -> tuple[List[str], List[int]]:
        """
        Generate responses with retry logic for both transient errors AND format validation.

        This method retries if:
        1. Network/JSON parse errors occur (transient issues)
        2. Format validation fails (format_score == -1)

        Args:
            prompts: List of prompts to send to model
            stage_name: Name of generation stage (for logging)
            max_tokens: Max tokens per response
            max_format_retries: Max retries for format failures (default 5)

        Returns:
            Tuple of (responses, format_scores)
            - responses: List of generated responses
            - format_scores: List of format scores (1=valid, -1=invalid)

        Raises:
            GenerationError: If final attempt fails
        """
        format_attempt = 0
        format_errors = 0

        while format_attempt < max_format_retries:
            try:
                # Generate with transient error retry logic
                responses = self._generate_with_retry(prompts, stage_name, max_tokens)

                # Validate format
                format_scores = [FormatValidator.validate_response(r)[0] for r in responses]

                # If all valid, return immediately
                if all(score == 1 for score in format_scores):
                    logger.info(f"[{stage_name}] Format validation passed on attempt {format_attempt + 1}")
                    return responses, format_scores

                # If some invalid, log and retry
                invalid_count = sum(1 for s in format_scores if s == -1)
                format_errors += invalid_count
                format_attempt += 1

                logger.warning(
                    f"[{stage_name}] Format validation failed: {invalid_count} invalid "
                    f"(attempt {format_attempt}/{max_format_retries})"
                )

                if format_attempt < max_format_retries:
                    # Wait before retry (exponential backoff)
                    time.sleep(self.retry_wait * format_attempt)
                    continue
                else:
                    # Final attempt failed, return what we have
                    logger.warning(
                        f"[{stage_name}] Format validation failed after {max_format_retries} attempts. "
                        f"Returning {invalid_count} invalid responses."
                    )
                    return responses, format_scores

            except JSONParseError as e:
                logger.warning(f"[{stage_name}] JSON parse error in format retry: {e}")
                format_attempt += 1
                if format_attempt < max_format_retries:
                    time.sleep(self.retry_wait * format_attempt)
                    continue
                else:
                    raise

            except NetworkError as e:
                logger.warning(f"[{stage_name}] Network error in format retry: {e}")
                format_attempt += 1
                if format_attempt < max_format_retries:
                    time.sleep(self.retry_wait * format_attempt)
                    continue
                else:
                    raise

        raise SkippableError(
            f"Failed to generate valid format for {stage_name} after {max_format_retries} attempts"
        )

    def _generate_refinement_template(self,
                                      initial_query: str,
                                      domain_enum: DomainType,
                                      num_guidance: int = None) -> tuple[str, List[str]]:
        """
        Generate refinement template and select guidance items.

        Args:
            initial_query: Original query to refine
            domain_enum: Domain for context
            num_guidance: Number of guidance items to apply (0-6, randomly picked if None)

        Returns:
            Tuple of (template, applied_guidance_list)
        """
        # Randomly pick num_guidance from 0-6 if not specified
        if num_guidance is None:
            num_guidance = random.randint(0, 6)

        # Randomly select guidance items
        num_guidance = min(num_guidance, len(QUERY_REFINEMENT_GUIDANCE_POOL))
        applied_guidance = random.sample(QUERY_REFINEMENT_GUIDANCE_POOL, num_guidance)

        # Combine guidance into a single string
        guidance_text = " ".join(applied_guidance)

        # Generate refinement prompt template
        domain_key = domain_enum.value
        refinement_template = generate_refinement_prompt(
            query=initial_query,
            domain1=WRITING_DOMAINS[domain_key]['name'],
            domain2=WRITING_DOMAINS[domain_key]['name'],
            guidance=guidance_text
        )

        return refinement_template, applied_guidance

    def _generate_refined_query_with_model(self,
                                          initial_query: str,
                                          domain_enum: DomainType) -> tuple[str, str, List[str], int]:
        """
        Generate refined query by sending refinement template to model.
        Retries up to 5 times if format validation fails.

        Args:
            initial_query: Original query to refine
            domain_enum: Domain for context

        Returns:
            Tuple of (template, refined_query, guidance_applied, format_score)
            format_score: 1 if valid format, -1 if invalid (after 5 retries)

        Raises:
            GenerationError: If generation fails after retries
        """
        template, applied_guidance = self._generate_refinement_template(initial_query, domain_enum)

        try:
            # Generate with format validation and retry logic
            responses, format_scores = self._generate_with_format_retry(
                [template], "refinement", max_format_retries=5
            )

            if responses and responses[0]:
                response = responses[0]
                format_score = format_scores[0]

                # Extract query from the response
                _, parsed_json = FormatValidator.validate_response(response)
                try:
                    if parsed_json and isinstance(parsed_json, dict):
                        refined_query = parsed_json.get("query", str(response))
                    else:
                        refined_query = str(parsed_json) if parsed_json else response
                except (TypeError, AttributeError):
                    refined_query = response

                return template, refined_query, applied_guidance, format_score
            else:
                raise SkippableError("Empty response from model")
        except GenerationError as e:
            logger.error(f"Refinement generation failed: {e}")
            raise

    def _generate_criteria_template(self, query: str) -> str:
        """
        Generate the criteria template (prompt to send to model).

        Args:
            query: The writing query to evaluate

        Returns:
            Criteria generation prompt template
        """
        criteria_prompt = generate_criteria_prompt(query)
        return criteria_prompt

    def _generate_evaluation_criteria_with_model(self,
                                                refined_query: str) -> tuple[str, Optional[List[Dict[str, Any]]], int]:
        """
        Generate evaluation criteria by sending template to model.
        Retries up to 3 times if format validation fails (less strict since optional).

        Criteria generation is optional — if it fails, returns (template, None, score).

        Args:
            refined_query: The refined writing query

        Returns:
            Tuple of (template, criteria_list_or_None, format_score)
            format_score: 1 if valid format, -1 if invalid (after 3 retries)
        """
        template = self._generate_criteria_template(refined_query)

        try:
            # Generate with format validation and retry logic (3 retries since optional)
            responses, format_scores = self._generate_with_format_retry(
                [template], "criteria_generation", max_format_retries=3
            )

            if responses and responses[0]:
                response = responses[0]
                format_score = format_scores[0]

                # Validate format and extract
                _, parsed_json = FormatValidator.validate_response(response)

                # Validate that it's a list
                if parsed_json and isinstance(parsed_json, list):
                    return template, parsed_json, format_score
                else:
                    logger.warning(f"Criteria not a list: {type(parsed_json)}")
                    return template, None, -1
            else:
                logger.warning("Empty response for criteria generation")
                return template, None, -1
        except GenerationError as e:
            # Criteria is optional — log and continue
            logger.warning(f"Criteria generation failed (non-fatal): {e}")
            return template, None, -1

    def extract_requirements_from_query(self, query: str) -> QueryRequirements:
        """
        Extract requirements from query text (heuristic-based).

        In a real system, this would parse the LLM-generated criteria.
        For now, we create empty requirements that would be populated
        by the actual LLM responses.

        Args:
            query: Query text

        Returns:
            QueryRequirements object
        """
        requirements = QueryRequirements()

        # Simple heuristic: check for requirement keywords
        query_lower = query.lower()

        if any(word in query_lower for word in ['style', 'tone', 'formal', 'casual', 'friendly']):
            requirements.style = "Inferred from query"

        if any(word in query_lower for word in ['format', 'template', 'structure', 'outline']):
            requirements.format = "Inferred from query"

        if any(word in query_lower for word in ['word', 'length', 'character', 'page']):
            requirements.length = "Inferred from query"

        return requirements

    def generate_single_prompt(self,
                              domain_enum: Optional[DomainType] = None,
                              subdomain: Optional[str] = None) -> tuple[Optional[WritingPrompt], Optional[List[Dict[str, Any]]]]:
        """
        Generate a single WritingPrompt with full pipeline and model inference.

        Handles errors gracefully:
        - Initial query failure → skip prompt, return None
        - Refinement failure → skip prompt, return None
        - Criteria failure → continue without criteria (non-fatal)

        Args:
            domain_enum: Specific domain (if None, sampled randomly)
            subdomain: Specific subdomain (if None, sampled from domain)

        Returns:
            Tuple of (WritingPrompt or None, error_list)
            - WritingPrompt if successful
            - None if unrecoverable error during critical stages
            - error_list: List of error dicts from generation
        """
        # Sample domain/subdomain if not provided
        if domain_enum is None or subdomain is None:
            domain_enum, subdomain = self.sampler.sample_domain_subdomain_pair()

        self.prompt_counter += 1
        prompt_id = f"prompt_{self.prompt_counter:04d}"
        domain_key = domain_enum.value
        domain_name = WRITING_DOMAINS[domain_key]['name']

        prompt_errors = []

        try:
            # Step 1: Generate initial query with model
            try:
                initial_template, initial_query, initial_format_score = self._generate_initial_query_with_model(domain_enum, subdomain)
            except GenerationError as e:
                error_detail = {
                    "prompt_id": prompt_id,
                    "stage": "initial_query",
                    "type": "json_parse" if isinstance(e, JSONParseError) else "network" if isinstance(e, NetworkError) else "unknown",
                    "error": str(e),
                    "action": "skip"
                }
                prompt_errors.append(error_detail)
                return None, prompt_errors

            # Step 2: Apply refinement with model
            try:
                refinement_template, refined_query, applied_guidance, refined_format_score = self._generate_refined_query_with_model(
                    initial_query, domain_enum
                )
            except GenerationError as e:
                error_detail = {
                    "prompt_id": prompt_id,
                    "stage": "refinement",
                    "type": "json_parse" if isinstance(e, JSONParseError) else "network" if isinstance(e, NetworkError) else "unknown",
                    "error": str(e),
                    "action": "skip"
                }
                prompt_errors.append(error_detail)
                return None, prompt_errors

            # Step 3: Generate evaluation criteria with model (optional)
            criteria_template, evaluation_criteria, criteria_format_score = self._generate_evaluation_criteria_with_model(refined_query)
            if evaluation_criteria is None:
                # Non-fatal: log but continue
                prompt_errors.append({
                    "prompt_id": prompt_id,
                    "stage": "criteria_generation",
                    "type": "non_fatal",
                    "error": "Failed to parse criteria (continuing without criteria)",
                    "action": "continue"
                })

            # Step 4: Extract requirements from refined query
            requirements = self.extract_requirements_from_query(refined_query)

            # Create WritingPrompt dataclass with all templates and outputs
            writing_prompt = WritingPrompt(
                prompt_id=prompt_id,
                domain=domain_enum,
                domain_name=domain_name,
                subdomain=subdomain,
                initial_query_template=initial_template,
                refinement_template=refinement_template,
                criteria_template=criteria_template,
                initial_query=initial_query,
                refined_query=refined_query,
                initial_query_format_score=initial_format_score,
                refined_query_format_score=refined_format_score,
                criteria_format_score=criteria_format_score,
                requirements=requirements,
                evaluation_criteria=evaluation_criteria,
                num_criteria=self.num_criteria,
                language=self.language,
                guidance_applied=applied_guidance,
                seed=self.seed
            )

            return writing_prompt, prompt_errors

        except Exception as e:
            # Unexpected error
            error_detail = {
                "prompt_id": prompt_id,
                "stage": "unknown",
                "type": "unexpected",
                "error": str(e),
                "action": "skip"
            }
            prompt_errors.append(error_detail)
            logger.error(f"Unexpected error in generate_single_prompt: {e}", exc_info=True)
            return None, prompt_errors

    def generate_batch(self, num_prompts: int, batch_id: Optional[str] = None) -> PromptBatch:
        """
        Generate a batch of prompts with error tracking.

        Args:
            num_prompts: Number of prompts to attempt to generate
            batch_id: Optional batch identifier

        Returns:
            PromptBatch with generation_log tracking successes and failures
        """
        if batch_id is None:
            batch_id = f"batch_seed{self.seed}_{int(time.time())}"

        batch = PromptBatch(batch_id=batch_id)

        for i in range(num_prompts):
            prompt, errors = self.generate_single_prompt()
            batch.add_prompt(prompt, errors=errors)

            if prompt is not None:
                logger.info(f"✓ Generated prompt {i + 1}/{num_prompts}: {prompt.prompt_id}")
            else:
                logger.warning(f"✗ Failed to generate prompt {i + 1}/{num_prompts}: {errors}")

        # Log batch summary
        log = batch.generation_log
        logger.info(
            f"\n=== Batch Summary: {batch_id} ===\n"
            f"  Total attempted: {log['total_attempted']}\n"
            f"  Successfully generated: {log['total_generated']}\n"
            f"  Skipped: {log['skipped']}\n"
            f"  JSON parse failures: {log['json_parse_failures']}\n"
            f"  Network failures: {log['network_failures']}\n"
        )

        return batch


# =============================================================================
# Utility Functions
# =============================================================================

def save_batch_to_json(batch: PromptBatch, output_path: str):
    """Save prompt batch to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(batch.to_json())

    print(f"Batch saved to {output_path}")


def load_batch_from_json(json_path: str) -> PromptBatch:
    """Load prompt batch from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    batch = PromptBatch(
        batch_id=data['batch_id'],
        generation_log=data.get('generation_log', {})
    )

    for prompt_data in data['prompts']:
        domain_enum = DomainType(prompt_data['domain'])
        requirements = QueryRequirements(**{
            k: v for k, v in prompt_data['requirements'].items()
            if k in ['style', 'format', 'length']
        })

        prompt = WritingPrompt(
            prompt_id=prompt_data['prompt_id'],
            domain=domain_enum,
            domain_name=prompt_data['domain_name'],
            subdomain=prompt_data['subdomain'],
            initial_query_template=prompt_data.get('initial_query_template', ''),
            refinement_template=prompt_data.get('refinement_template', ''),
            criteria_template=prompt_data.get('criteria_template', ''),
            initial_query=prompt_data['initial_query'],
            refined_query=prompt_data['refined_query'],
            initial_query_format_score=prompt_data.get('initial_query_format_score', 1),
            refined_query_format_score=prompt_data.get('refined_query_format_score', 1),
            criteria_format_score=prompt_data.get('criteria_format_score', 1),
            requirements=requirements,
            evaluation_criteria=prompt_data.get('evaluation_criteria'),
            num_criteria=prompt_data.get('num_criteria', 5),
            language=prompt_data.get('language', 'English'),
            guidance_applied=prompt_data.get('guidance_applied', []),
            seed=prompt_data.get('seed', 42)
        )
        batch.add_prompt(prompt)

    return batch


# =============================================================================
# Example Usage & Testing (requires vLLM or ModelClient)
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Challenger Prompt Generation Pipeline - Example")
    print("=" * 80)
    print("\nNOTE: This example requires a ModelClient implementation (e.g., vLLM).")
    print("To run with vLLM, use Modal: modal run modal_run.py::generate_prompts")
    print("\nLocal testing without a model client is not supported (model_client is required).")
    print("=" * 80)





