"""
Question Generation Module for R-Zero Creative Writing Pipeline

This package contains:
- creative_writing_prompts: WritingBench prompt templates and domain definitions
- creative_question_generate: Pipeline for generating challenger prompts with vLLM inference
"""

from .creative_question_generate import (
    # Pipeline and data structures
    ChallengerPromptPipeline,
    WritingPrompt,
    PromptBatch,
    QueryRequirements,
    DomainType,
    # Model client interface
    ModelClient,
    # Error classes
    GenerationError,
    JSONParseError,
    NetworkError,
    SkippableError,
    # Utilities
    save_batch_to_json,
    load_batch_from_json,
)

__all__ = [
    # Pipeline
    "ChallengerPromptPipeline",
    # Data structures
    "WritingPrompt",
    "PromptBatch",
    "QueryRequirements",
    "DomainType",
    # Model client
    "ModelClient",
    # Errors
    "GenerationError",
    "JSONParseError",
    "NetworkError",
    "SkippableError",
    # Utilities
    "save_batch_to_json",
    "load_batch_from_json",
]
