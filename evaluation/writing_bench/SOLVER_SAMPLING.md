# Solver Sampling Pipeline

Generate M samples per prompt and compute R-Zero uncertainty reward.

## Overview

**Pipeline Flow:**
```
one_shot.json (from one_shot_creative_question_generate.py)
    ↓
SolverSampler (M samples per prompt)
    ↓ vLLM batch generation
    ↓
SampleEvaluator (evaluate all M samples)
    ↓ Claude/Critic evaluation
    ↓
WritingPromptResponse (raw data + statistics)
    ↓
RewardComputation (R-Zero uncertainty)
    ↓ r = 1 - 2|mean/10 - 0.5|
    ↓
Output: responses.jsonl + rewards_summary.json
```

## Quick Start

### 1. Generate questions first
```bash
python question_generate/one_shot_creative_question_generate.py \
    --model Qwen/Qwen3-4B \
    --num_samples 10 \
    --suffix test
```
Output: `$STORAGE_PATH/generated_questions_one_shot/one_shot_writingbench_test.json`

### 2. Run solver sampling (M=5 samples per prompt)
```bash
python evaluation/writing_bench/solver_sampling.py \
    --model Qwen/Qwen3-4B \
    --input_json creative_generated_samples/one_shot.json \
    --output_dir evaluation/writing_bench/solver_sampling_outputs/run_20250515_m5 \
    --M 5 \
    --evaluator claude
```

**Output files:**
- `responses.jsonl` — All WritingPromptResponse data (M samples per prompt)
- `rewards_summary.json` — Aggregated statistics + top/bottom rewards

### 3. Inspect results
```bash
# View first response
head -1 evaluation/writing_bench/solver_sampling_outputs/run_20250515_m5/responses.jsonl | jq .

# View rewards summary
cat evaluation/writing_bench/solver_sampling_outputs/run_20250515_m5/rewards_summary.json | jq .
```

## R-Zero Uncertainty Reward

**Formula (from R-Zero paper):**
```
r_uncertainty(x; φ) = 1 - 2|p̂(x; S_φ) - 1/2|
```

Where:
- `p̂(x; S_φ)` = average score across M samples (normalized to [0, 1])
- For WritingBench (1-10 scale): `r = 1 - 2|mean_score/10 - 0.5|`

**Interpretation:**
- **r = 1.0** → Mean score = 5.0 (solver completely confused, maximum learning potential)
- **r = 0.8** → Mean score = 4.0 or 6.0 (still some uncertainty)
- **r = 0.4** → Mean score = 2.0 or 8.0 (solver confident, low learning potential)
- **r = 0.0** → Mean score = 1.0 or 10.0 (solver certain, no learning value)

## WritingPromptResponse Data Structure

See `evaluation/shared/data_models.py`

**Key fields:**
```python
class WritingPromptResponse:
    prompt_id: str
    query: str
    criteria: List[Dict]
    
    # M sampled responses
    samples: List[SampleResult]  # [SampleResult(sample_id, text, scores), ...]
    M: int  # Number of samples
    
    # Aggregated statistics
    statistics: Dict[str, Dict]  # {criterion: {mean, std, min, max, ...}}
    
    # R-Zero uncertainty reward
    reward: float  # In [0, 1]
```

## Example Output

### responses.jsonl (one per line)
```json
{
  "prompt_id": "prompt_0001",
  "domain": "D6",
  "domain_name": "Advertising & Marketing",
  "subdomain": "Product Description",
  "query": "Craft a compelling...",
  "M": 5,
  "samples": [
    {
      "sample_id": 0,
      "text": "This product represents...",
      "scores": {
        "Specificity and Detail": {"score": 7, "reason": "..."},
        "Tone and Voice": {"score": 6, "reason": "..."}
      }
    },
    ... (4 more samples)
  ],
  "statistics": {
    "Specificity and Detail": {
      "mean": 6.8,
      "std": 1.1,
      "min": 5,
      "max": 8,
      "percentile_25": 6.0,
      "percentile_75": 7.5
    },
    ...
  },
  "reward": 0.636
}
```

### rewards_summary.json
```json
{
  "num_prompts": 10,
  "M": 5,
  "reward_mean": 0.54,
  "reward_std": 0.21,
  "reward_min": 0.08,
  "reward_max": 0.92,
  "score_mean": 5.8,
  "score_std": 1.5,
  "top_5_by_reward": [
    {"prompt_id": "prompt_0042", "reward": 0.92, "score": 5.1},
    {"prompt_id": "prompt_0015", "reward": 0.88, "score": 5.4},
    ...
  ],
  "bottom_5_by_reward": [
    {"prompt_id": "prompt_0003", "reward": 0.08, "score": 8.9},
    ...
  ]
}
```

## Exploring Results

### Top learnable prompts (highest uncertainty reward)
```python
import json

with open("rewards_summary.json") as f:
    summary = json.load(f)
    
top_5 = summary["top_5_by_reward"]
print("Top 5 learnable prompts:")
for item in top_5:
    print(f"  {item['prompt_id']}: reward={item['reward']:.3f}, score={item['score']:.1f}")
```

### Load and inspect a prompt response
```python
from evaluation.shared import WritingPromptResponse

with open("responses.jsonl") as f:
    response_data = json.loads(f.readline())
    
response = WritingPromptResponse.from_dict(response_data)
print(f"Prompt: {response.prompt_id}")
print(f"Reward: {response.reward:.3f}")
print(f"M samples: {response.M}")
print(f"Aggregate score: {response.compute_aggregate_score():.1f}")

# View per-criterion stats
for criterion, stats in response.statistics.items():
    print(f"  {criterion}: mean={stats['mean']:.1f} std={stats['std']:.2f}")
```

## Parameters

### Generation (vLLM)
- `--temperature` (default 0.7) — Higher = more diverse samples
- `--top_p` (default 0.8) — Nucleus sampling
- `--top_k` (default 20) — Top-k sampling
- `--max_tokens` (default 16000) — Max response length

### Sampling
- `--M` (default 5) — Number of samples per prompt
  - M=3: Quick initial run (~30 min for 50 prompts)
  - M=5: Recommended (~50 min for 50 prompts)
  - M=10: Higher variance reduction (~100 min for 50 prompts)

### Evaluation
- `--evaluator` — "claude" (recommended) or "critic"
  - Claude: More reliable, uses Claude API
  - Critic: Faster, uses local critic model

## Troubleshooting

### Out of memory
- Reduce batch size (not directly supported; modify solver_sampling.py)
- Use smaller M
- Use smaller model

### Evaluation timeouts
- Use `--evaluator critic` (faster)
- Reduce number of criteria in one_shot.json

### Low reward prompts
- Check if prompts are too easy/hard for model
- Consider adjusting generation temperature
- Review one_shot_creative_question_generate.py criteria design

## Next Steps: R-Zero Integration

Use the rewards to select prompts for improvement:

```python
# Load responses
responses = []
with open("responses.jsonl") as f:
    for line in f:
        data = json.loads(line)
        responses.append(WritingPromptResponse.from_dict(data))

# Select high-uncertainty prompts
learnable = sorted(
    responses,
    key=lambda r: r.reward,
    reverse=True
)[:10]

print(f"Selected {len(learnable)} prompts to improve:")
for r in learnable:
    print(f"  {r.prompt_id}: reward={r.reward:.3f}")

# Now apply refinements to these prompts
# and re-run solver_sampling to measure improvement
```

## References

- **R-Zero paper:** [Link to paper]
- **WritingBench:** https://github.com/X-PLUG/WritingBench
- **vLLM:** https://docs.vllm.ai/
