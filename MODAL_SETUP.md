# Modal Setup Guide for Qwen Pipeline

## Quick Start (5 minutes)

### 1. Install Modal
```bash
pip install modal
```

### 2. Authenticate
```bash
modal token new
```
This opens a browser to create a free Modal account. Follow the steps and paste the token.

### 3. Run Single Example (Template Mode - No Inference)
```bash
cd /Users/mertaydayanc/Desktop/Columbia/daplab/R-zero-testing/question_generate
python modal_pipeline.py --num-prompts 1
```

Output shows the prompt templates that will be sent to Qwen (no actual inference yet).

### 4. Run with Qwen Inference
```bash
modal run modal_pipeline.py --num-prompts 1 --modal
```

This:
- Downloads Qwen/Qwen3-4B-Base (cached for future runs)
- Allocates A40 GPU
- Runs inference
- Takes ~1-2 minutes total

## Workflow for Iteration

### Fast Iteration Loop:
1. **Modify** `creative_question_generate.py` or `creative_writing_prompts.py`
2. **Test locally** with templates:
   ```bash
   python modal_pipeline.py --num-prompts 1
   ```
3. **Once validated**, test with actual inference:
   ```bash
   modal run modal_pipeline.py --num-prompts 1 --modal
   ```

### Save Results:
```bash
modal run modal_pipeline.py --num-prompts 3 --modal --save results.json
```

## Understanding the Pipeline

```
Local (Your MacBook)          →    Modal Cloud (GPU)
┌─────────────────────────┐       ┌──────────────────┐
│ Generate WritingPrompts │       │ Load Qwen Model  │
│ (dataclasses)           │  →    │ Run Inference    │
│                         │       │ Return Generated │
│ Submit to Modal         │       │ Text             │
└─────────────────────────┘       └──────────────────┘
```

## Cost & Limits

- **Free Modal Tier**: 
  - ~100 free GPU hours/month
  - A40 GPU available
  - Sufficient for iteration & validation
  
- **Cost**: ~$0.30-0.50 per 1000 inference calls with A40
  - One single-prompt run = ~1 cent

## Troubleshooting

### "ImportError: No module named 'creative_question_generate'"
Make sure you're running from the correct directory:
```bash
cd /Users/mertaydayanc/Desktop/Columbia/daplab/R-zero-testing/question_generate
```

### "CUDA out of memory"
Modal auto-allocates larger GPU if needed. Usually A40 is sufficient for 4B model.

### "Model download times out"
First run downloads the model (~10GB). If it times out:
```bash
# Pre-download with longer timeout
modal run modal_pipeline.py --num-prompts 1 --modal
# (This caches it; next runs will be faster)
```

## Next Steps

Once you've validated the pipeline:

1. **Scale up**: Change `num_prompts` from 1 → 10 → 100
2. **Parse outputs**: The generated text is currently just printed. Add JSON parsing for structured extraction
3. **Production**: When ready, move to distributed sampling with `modal.Queue` for large-scale generation

## Example Output

```
================================================================================
Challenger Prompt Generation Pipeline (with Qwen)
================================================================================

[Step 1] Generating 1 prompt(s) locally...
✓ Generated batch: batch_000 with 1 prompts

[Step 2] Enhancing prompts with Qwen...
================================================================================
Processing Prompt 1/1: prompt_0001
  Domain: Literature & Arts > Poetry
  Guidance: ['Add style requirements...', 'Add a requirement for generating...']

  Running initial_query generation through Qwen...
  ✓ Generated (first 100 chars): Write a haiku about the passage of time, 
  using metaphor and natural imagery. Ensure the poem captures both the ...
```
