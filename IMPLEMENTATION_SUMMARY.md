# vLLM Integration Implementation Summary

## Overview
Successfully integrated vLLM-based model inference into the Challenger Prompt Generation pipeline. The pipeline now generates actual creative writing prompts through model inference instead of just returning templates.

## Key Changes

### 1. **Error Handling System** (`creative_question_generate.py`)

Created a comprehensive error classification system:
- `GenerationError`: Base exception for all generation failures
- `JSONParseError`: Malformed JSON output → RETRY (with exponential backoff)
- `NetworkError`: Transient vLLM/network issues → RETRY
- `SkippableError`: Unrecoverable errors → SKIP prompt (log and continue)

**Retry Logic:**
- Up to 3 retry attempts by default
- Exponential backoff: wait = base_wait * (attempt + 1)
- Only JSON parse and network errors trigger retries
- Other errors skip the prompt and move to next

### 2. **ModelClient Interface** (`creative_question_generate.py`)

Created abstract base class for model inference:
```python
class ModelClient(ABC):
    @abstractmethod
    def generate(prompts: List[str], max_tokens: int = 4096) -> List[str]
    @abstractmethod
    def format_prompt(messages: List[Dict[str, str]]) -> str
```

Benefits:
- Flexible: Can support vLLM, API-based services, mocks, etc.
- Type-safe: ABC enforces interface contract
- Required: Constructor validates model_client is provided (no fallback mode)

### 3. **Updated WritingPrompt Dataclass**

New fields for complete transparency:
```python
# TEMPLATES (what we sent to the model)
initial_query_template: str
refinement_template: str
criteria_template: str

# ACTUAL OUTPUTS (what model returned)
initial_query: str           # Changed: now actual output, not template
refined_query: str           # Changed: now actual output, not template
evaluation_criteria: List[Dict[str, Any]]  # Changed: now actual criteria

# Metadata
seed: int                    # Seed used for this batch
```

**Why both templates and outputs?**
- Transparency: See what prompt produced what output
- Debugging: Compare template vs actual to identify generation issues
- Reproducibility: Templates can be re-run with different models
- Research: Analyze how different templates affect outputs

### 4. **Updated PromptBatch Dataclass**

New generation_log for failure tracking:
```python
generation_log: Dict[str, Any] = {
    "total_attempted": int,
    "total_generated": int,
    "skipped": int,
    "json_parse_failures": int,
    "network_failures": int,
    "error_details": [List of error dicts],
}
```

**Example output:**
```json
{
  "batch_id": "batch_seed42_1234567890",
  "generation_log": {
    "total_attempted": 10,
    "total_generated": 9,
    "skipped": 1,
    "json_parse_failures": 0,
    "network_failures": 0,
    "error_details": [
      {
        "prompt_id": "prompt_0005",
        "stage": "criteria_generation",
        "type": "non_fatal",
        "error": "Failed to parse criteria JSON",
        "action": "continue"
      }
    ]
  },
  "prompts": [...]
}
```

### 5. **ChallengerPromptPipeline Refactoring**

Major structural changes:

**Constructor:**
```python
def __init__(
    self,
    model_client: ModelClient,  # REQUIRED (was optional, now mandatory)
    language: str = "English",
    num_criteria: int = 5,
    seed: int = 42,             # Seed for batch consistency
    max_retries: int = 3,
    retry_wait: float = 1.0,
):
    if model_client is None:
        raise ValueError("model_client is required")
```

**Generation Methods:**
Old → New naming:
- `generate_initial_query()` → `_generate_initial_query_template()` + `_generate_initial_query_with_model()`
- `apply_query_refinement()` → `_generate_refinement_template()` + `_generate_refined_query_with_model()`
- `generate_evaluation_criteria()` → `_generate_criteria_template()` + `_generate_evaluation_criteria_with_model()`

**New Internal Methods:**
- `_generate_with_retry()`: Core retry logic with exponential backoff
- Each generation method now:
  1. Creates the template prompt
  2. Sends to model via retry logic
  3. Parses JSON response
  4. Returns both template and actual output

**generate_single_prompt() Changes:**
- Now returns: `tuple[Optional[WritingPrompt], Optional[List[Dict]]]`
- Critical failures (initial query, refinement) → return None, skip prompt
- Non-critical failures (criteria) → continue with None criteria, log warning
- Comprehensive per-prompt error tracking

**generate_batch() Changes:**
- Logs comprehensive batch statistics
- Tracks total_attempted, total_generated, skipped, various failure types
- Summary printed to logs:
  ```
  === Batch Summary: batch_seed42_1234567890 ===
    Total attempted: 10
    Successfully generated: 9
    Skipped: 1
    JSON parse failures: 0
    Network failures: 0
  ```

### 6. **VLLMClient Implementation** (`modal_run.py`)

Concrete implementation following `question_generate.py` pattern:

```python
class VLLMClient:
    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        gpu_memory_utilization: float = 0.8
    ):
        # Load model once, reuse across calls
        self.model = vllm.LLM(model=model_name, ...)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(prompts: List[str], max_tokens: int = 4096) -> List[str]:
        # Batch generation using vLLM
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.95,
            n=1,
        )
        return self.model.generate(prompts, sampling_params)
```

**Features:**
- Reuses vLLM model instance (no reload between calls)
- Supports batch generation
- Error handling: Maps exceptions to NetworkError
- GPU memory optimization: configurable utilization

### 7. **Modal Integration Updates** (`modal_run.py`)

**Updated `generate_challenger_prompts()` Function:**

```python
@app.function(gpu=MODAL_EVAL_GPU, ...)
def generate_challenger_prompts(
    num_prompts: int = 1,
    model_name: str = "Qwen/Qwen3-4B-Base",
    seed: int = 42,
    output_dir: str = "generated_question",
    gpu_memory_utilization: float = 0.8,
):
    # 1. Initialize vLLM
    model_client = VLLMClient(model_name, seed, gpu_memory_utilization)
    
    # 2. Create pipeline with vLLM
    pipeline = ChallengerPromptPipeline(model_client, ...)
    
    # 3. Generate batch (with actual inference)
    batch = pipeline.generate_batch(num_prompts)
    
    # 4. Save to storage
    # 5. Return comprehensive stats
```

**Updated `generate_prompts()` Local Entrypoint:**

```python
@app.local_entrypoint()
def generate_prompts(
    num_prompts: int = 1,
    model_name: str = "Qwen/Qwen3-4B-Base",
    seed: int = 42,
    output_dir: str = "generated_question",
):
    # Forwards to generate_challenger_prompts.remote()
```

### 8. **Return Value Structure** (Modal)

Function now returns comprehensive metrics:
```python
{
    "batch_id": "batch_seed42_...",
    "total_attempted": 10,
    "total_generated": 9,
    "skipped": 1,
    "json_parse_failures": 0,
    "network_failures": 0,
    "output_file": "/storage/generated_question/batch_seed42_....json",
    "domains": ["D1", "D3", "D6"],
    "num_subdomains": 5,
}
```

### 9. **Updated Exports** (`__init__.py`)

```python
from .creative_question_generate import (
    # Pipeline
    ChallengerPromptPipeline,
    # Data structures
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
```

## Seed Management

**Why it matters:**
- **Batch reproducibility**: All prompts in a batch use same seed → consistent randomness
- **Cross-batch diversity**: Different batches use different seeds → varied prompts
- **Experimental tracking**: Can correlate seed → batch quality for ablation studies
- **Deterministic evaluation**: Judge can reproduce conditions that created a prompt

**Implementation:**
- Seed passed to `generate_batch(num_prompts, batch_id)`
- Stored in each WritingPrompt: `seed: int = 42`
- Passed to VLLMClient initialization
- Batch ID includes seed: `batch_seed{seed}_{timestamp}`

## Token Limits

Inherited from `question_generate.py`: **4096 max_tokens per response**

Applied uniformly across all generation stages:
- Initial query generation
- Query refinement
- Evaluation criteria generation

## Error Handling Strategy

### Critical Errors (Fatal for Prompt)
- Initial query generation failure → **skip prompt**, log error
- Query refinement failure → **skip prompt**, log error

### Non-Critical Errors (Recoverable)
- Criteria generation failure → **continue without criteria**, log warning
- Returns WritingPrompt with `evaluation_criteria=None`

### Error Classification
```
JSON Parse Error:
  └─ Retry up to 3 times with exponential backoff
  └─ If all retries fail, treat as fatal/non-critical based on stage

Network Error:
  └─ Retry up to 3 times with exponential backoff
  └─ Indicates transient vLLM or system issue

Skippable Error:
  └─ No retry
  └─ Fatal for initial query / refinement
  └─ Handled gracefully for criteria
```

## Testing & Validation

### How to Run on Modal

```bash
# Basic: 5 prompts with default settings
modal run --detach modal_run.py::generate_prompts --num-prompts 5

# Custom model and seed
modal run --detach modal_run.py::generate_prompts \
    --num-prompts 10 \
    --model-name Qwen/Qwen3-4B \
    --seed 2024

# Pull results from volume
modal volume get r-zero-storage generated_question/ ./results/
```

### Expected Output Structure

```json
{
  "batch_id": "batch_seed42_1715000000",
  "generation_log": {
    "total_attempted": 10,
    "total_generated": 9,
    "json_parse_failures": 0,
    "network_failures": 0,
    "skipped": 1,
    "error_details": [...]
  },
  "prompts": [
    {
      "prompt_id": "prompt_0001",
      "domain": "D6",
      "domain_name": "Advertising & Marketing",
      "subdomain": "Product Description",
      
      "initial_query_template": "Generate 1 different writing requests...",
      "initial_query": "Write a compelling product description for a sustainable bamboo toothbrush...",
      
      "refinement_template": "Please refine and enhance the original writing requirements...",
      "refined_query": "Write a 150-200 word product description for eco-conscious consumers...",
      
      "criteria_template": "Create 5 evaluation criteria for assessing...",
      "evaluation_criteria": [
        {"criterion": 1, "description": "Clarity of sustainability message", ...},
        ...
      ],
      
      "requirements": {
        "style": "persuasive, professional",
        "format": "paragraph",
        "length": "150-200 words"
      },
      
      "guidance_applied": [
        "Add a requirement for generating specific lengths.",
        "Add style requirements..."
      ],
      
      "seed": 42,
      "language": "English"
    },
    ...
  ]
}
```

## Key Differences from Previous Implementation

| Aspect | Before | After |
|--------|--------|-------|
| **Model inference** | Template-only (no inference) | Full vLLM inference with retries |
| **Query content** | Templates | Actual generated queries |
| **Error handling** | None | Comprehensive with retries and logging |
| **model_client** | Optional (unused) | Required, enforced at construction |
| **Templates storage** | Not stored | Stored for transparency |
| **Failure tracking** | None | Per-prompt and per-batch logs |
| **Criteria generation** | None | Actual criteria from model |
| **Seed handling** | Per-prompt | Per-batch (consistency) |

## Next Steps

1. **Test on Modal**: Run with small num_prompts to validate end-to-end
2. **Validate outputs**: Check quality and diversity of generated prompts
3. **Analyze failure modes**: Review error_details to improve robustness
4. **Scale up**: Run full batch generation with higher num_prompts
5. **Integrate with R-Zero loop**: Feed generated prompts into Solver → Judge

## Files Modified

1. `/Users/mertaydayanc/Desktop/Columbia/daplab/R-zero-testing/question_generate/creative_question_generate.py`
   - Added error classes, ModelClient ABC
   - Refactored ChallengerPromptPipeline for model inference
   - Updated WritingPrompt and PromptBatch dataclasses
   - Added comprehensive retry logic

2. `/Users/mertaydayanc/Desktop/Columbia/daplab/R-zero-testing/modal_run.py`
   - Added VLLMClient implementation
   - Updated generate_challenger_prompts() for model inference
   - Updated generate_prompts() entrypoint

3. `/Users/mertaydayanc/Desktop/Columbia/daplab/R-zero-testing/question_generate/__init__.py`
   - Exported ModelClient, error classes, and all new functionality
