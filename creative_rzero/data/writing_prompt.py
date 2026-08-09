"""creative_rzero/data/writing_prompt.py — the WritingPrompt data model.

Shared by prompt generation (`creative_rzero/steps/generate_prompts.py`,
which produces these), the parquet builder (`steps/build_parquet.py`, which
serializes them into training rows), and both reward callers
(`examples/reward_function/creative_*_caller.py`, which deserialize them
back out of the parquet's `answer` column to recover query/criteria at
reward time). Split out from the generation logic so all of those can
import it without pulling in vllm/transformers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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
    # Raw <think>…</think> reasoning trace the model produced before the answer
    # (empty for non-thinking models). Stored for inspection, not used downstream.
    thinking: str = ""

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
            thinking=data.get("thinking", ""),
        )


@dataclass
class PromptBatch:
    """Collection of prompts generated in a batch."""

    batch_id: str
    prompts: List[WritingPrompt] = field(default_factory=list)
    domains_sampled: List[str] = field(default_factory=list)
    subdomains_sampled: List[str] = field(default_factory=list)
    generation_log: Dict[str, Any] = field(default_factory=dict)
    # Per-attempt failure records (reason + full raw response) from the
    # generation retry loop. Deliberately excluded from to_dict(): the raw
    # responses would bloat the prompts JSON, so run_generation writes them
    # to a `.failures.jsonl` sidecar instead. Aggregate counts live in
    # generation_log["failure_reason_counts"].
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.generation_log:
            self.generation_log = {
                "total_attempted": 0,
                "total_generated": 0,
                "skipped": 0,
                "json_parse_failures": 0,
                "network_failures": 0,
                "format_validation_failures": 0,
                "language_filter_failures": 0,
                "failure_reason_counts": {},
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
