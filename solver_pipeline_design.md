# Solver Pipeline Architecture Design
## R-Zero Creative Task Solver with Pairwise Ranking Rewards

**Date:** 2026-05-25  
**Status:** Design Phase  
**Sample Scale:** 8-16 samples per iteration (smoke test configuration)

---

## 1. System Overview

The solver pipeline creates a closed-loop system where:
1. **Challenger** generates improved prompts (checkpoint output)
2. **Solver** samples outputs for those prompts
3. **Evaluator** scores the outputs
4. **Reward Assigner** computes normalized rank scores for GRPO training

```
┌─────────────────────────────────────────────────────────────┐
│                    SOLVER PIPELINE FLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Challenger Checkpoint                                        │
│        │                                                      │
│        └──> CheckpointLoader                                 │
│             │                                                 │
│             └──> PromptGenerator (@question_generate)        │
│                  │                                            │
│                  └──> SolverSampler                           │
│                       │                                       │
│                       └──> BatchEvalAgent                     │
│                            │                                  │
│                            └──> RewardAssigner                │
│                                 (Normalized Rank)            │
│                                 │                             │
│                                 └──> GRPO Training Data       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Checkpoint Loader

**Purpose:** Load challenger checkpoint and validate integrity

**Input:**  
- `checkpoint_path`: `/storage/models/qwen3-4b-creative-smoke_challenger_v1/`

**Output:**  
- `loaded_model`: Ready-to-use model instance
- `metadata`: Training metadata, iteration count, etc.

**Implementation Notes:**
- Support multiple formats (defer format decision with factory pattern):
  - HuggingFace safetensors + config
  - PyTorch checkpoint (.pt)
  - Full directory structure
- Validate checkpoint integrity (check required files)
- Load metadata to track lineage and version

```python
class CheckpointLoader:
    def __init__(self, supported_formats: List[str] = None):
        """
        Args:
            supported_formats: ['huggingface', 'pytorch', 'directory']
        """
        self.formats = supported_formats or ['huggingface', 'pytorch', 'directory']
    
    def detect_format(self, path: str) -> str:
        """Auto-detect checkpoint format"""
        pass
    
    def load(self, checkpoint_path: str) -> Tuple[torch.nn.Module, Dict]:
        """Load and return (model, metadata)"""
        pass
    
    def validate(self, checkpoint_path: str) -> bool:
        """Verify checkpoint integrity"""
        pass
```

---

### 2.2 Prompt Generator

**Purpose:** Generate creative prompts from challenger checkpoint using one-shot learning

**Input:**  
- Challenger checkpoint (via CheckpointLoader)
- Configuration: number of prompts to generate (N=8-16)

**Output:**  
- `prompts`: List[str] of creative prompts

**Integration:**
- Wraps `@question_generate/one_shot_creative_question_generate.py`
- Could be prompt-based (zero-shot template) or model-based (sample from challenger)
- For smoke test: consider using fixed/templated prompts to ensure reproducibility

```python
class PromptGenerator:
    def __init__(self, 
                 checkpoint_path: str,
                 generator_module_path: str = "@question_generate/one_shot_creative_question_generate.py",
                 num_prompts: int = 16):
        self.checkpoint = CheckpointLoader().load(checkpoint_path)
        self.num_prompts = num_prompts
        self.generator = self._import_generator(generator_module_path)
    
    def generate_batch(self, 
                      temperature: float = 0.7,
                      top_k: int = 50) -> List[str]:
        """Generate N prompts"""
        pass
    
    def _import_generator(self, module_path: str):
        """Dynamically import generator module"""
        pass
```

---

### 2.3 Solver Sampler

**Purpose:** Generate multiple outputs (solver responses) for each prompt

**Input:**  
- `prompts`: List[str]
- `solver_model`: Current solver instance
- `num_samples_per_prompt`: M (typically 1 for smoke test, could be >1 for diversity)

**Output:**  
- `samples`: Dict mapping `prompt_id` → List[str] of M outputs

**Key Design Decisions:**
- Support multiple sampling strategies (temperature, nucleus sampling, etc.)
- Store full prompt-output pairs for traceability
- Seed for reproducibility in smoke tests

```python
class SolverSampler:
    def __init__(self, 
                 solver_model: torch.nn.Module,
                 device: str = "cuda",
                 seed: int = 42):
        self.model = solver_model
        self.device = device
        self.seed = seed
    
    def sample_batch(self,
                    prompts: List[str],
                    num_samples_per_prompt: int = 1,
                    max_length: int = 512,
                    temperature: float = 0.8,
                    top_p: float = 0.95) -> Dict[int, List[str]]:
        """
        Args:
            prompts: List of input prompts
            num_samples_per_prompt: How many outputs per prompt
        
        Returns:
            {prompt_id: [output1, output2, ...]}
        """
        pass
    
    def _set_seed(self):
        """Ensure reproducible sampling"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
```

---

### 2.4 Evaluation Component

**Purpose:** Score solver outputs using existing BatchEvalAgent

**Input:**  
- `prompt_sample_pairs`: List[Tuple[prompt, output]]
- Evaluation criteria configuration

**Output:**  
- `scores`: Dict mapping `(prompt_id, sample_id)` → float score ∈ [0, 1]

**Design:**
- Assume BatchEvalAgent already handles the scoring logic
- Batch prompts and outputs for efficiency
- Store scores with full traceability (which prompt, which solver, which sample)

```python
class EvaluationPipeline:
    def __init__(self, batch_eval_agent):
        """
        Args:
            batch_eval_agent: Existing BatchEvalAgent instance
        """
        self.evaluator = batch_eval_agent
    
    def evaluate_samples(self,
                        prompt_sample_pairs: List[Tuple[str, str]],
                        batch_size: int = 8) -> Dict[Tuple[int, int], float]:
        """
        Args:
            prompt_sample_pairs: [(prompt, output), ...]
        
        Returns:
            {(prompt_id, sample_id): score}
        """
        pass
```

---

### 2.5 Reward Assignment (Pairwise Tournament with Normalized Rank)

**Purpose:** Compute normalized rank-based rewards for GRPO training

**Mathematical Formulation:**

For each prompt with M outputs:
1. Rank outputs by evaluation score (1 = best, M = worst)
2. Compute normalized score: 
   ```
   R_i = (G - rank_i) / (G - 1)
   
   where:
   - G = total number of samples (e.g., 16)
   - rank_i ∈ [1, G]
   - R_i ∈ [0, 1]
   ```

**Properties:**
- Mean reward = 0.5 (half positive advantage, half negative)
- Sample efficient for GRPO training
- Captures relative ranking without absolute score dependency

**Example:**
```
16 outputs ranked by BatchEvalAgent scores:
Rank 1 (best):  R = (16-1)/(16-1) = 15/15 = 1.0   (max advantage)
Rank 8:         R = (16-8)/(16-1) = 8/15  = 0.533  (slight advantage)
Rank 9:         R = (16-9)/(16-1) = 7/15  = 0.467  (slight disadvantage)
Rank 16 (worst):R = (16-16)/(16-1) = 0/15 = 0.0   (max disadvantage)

Mean = (sum of R_i) / 16 = 8/16 = 0.5 ✓
```

**Implementation:**

```python
class RewardAssigner:
    def __init__(self, total_samples: int = 16):
        """
        Args:
            total_samples: G parameter for normalization
        """
        self.G = total_samples
    
    def compute_normalized_rank_rewards(self,
                                        eval_scores: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """
        Args:
            eval_scores: {(prompt_id, sample_id): score} from BatchEvalAgent
        
        Returns:
            rewards: {(prompt_id, sample_id): normalized_rank_reward}
        
        Algorithm:
            1. Group scores by prompt_id
            2. For each prompt group:
               a. Sort samples by score (descending)
               b. Assign ranks [1, 2, ..., M]
               c. Compute R_i = (G - rank_i) / (G - 1)
            3. Return all (prompt_id, sample_id) → R_i mappings
        """
        rewards = {}
        
        # Group by prompt
        prompt_groups = self._group_by_prompt(eval_scores)
        
        for prompt_id, samples in prompt_groups.items():
            # Sort by score descending
            sorted_samples = sorted(samples.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)
            
            # Assign normalized ranks
            for rank, (sample_id, score) in enumerate(sorted_samples, start=1):
                normalized_reward = (self.G - rank) / (self.G - 1)
                rewards[(prompt_id, sample_id)] = normalized_reward
        
        return rewards
    
    def _group_by_prompt(self, scores: Dict) -> Dict[int, Dict[int, float]]:
        """Reorganize {(pid, sid): score} to {pid: {sid: score}}"""
        pass
    
    def get_statistics(self, rewards: Dict) -> Dict:
        """Return mean, std, min, max of rewards for validation"""
        pass
```

---

## 3. Data Flow and Structures

### 3.1 Sample Structure

Each complete sample for GRPO training:

```python
@dataclass
class SolverSample:
    """Complete data for one solver output"""
    prompt_id: int
    sample_id: int
    prompt: str                          # Generated by PromptGenerator
    output: str                          # Generated by Solver
    evaluation_score: float              # From BatchEvalAgent [0, 1]
    normalized_rank_reward: float        # From RewardAssigner [0, 1]
    checkpoint_version: str              # Challenger version
    timestamp: float                     # For lineage tracking
    metadata: Dict[str, Any]             # Flexible metadata
```

### 3.2 Pipeline Output Format

```python
@dataclass
class SolverPipelineOutput:
    """Complete pipeline execution result"""
    run_id: str                          # Unique identifier
    timestamp: float
    checkpoint_path: str
    prompts: List[str]                   # N prompts
    samples: List[SolverSample]          # N prompts × M samples
    reward_statistics: Dict[str, float]  # mean, std, min, max
    
    def save(self, output_dir: str):
        """Save as JSON/pickle for GRPO consumption"""
        pass
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Check data integrity and constraints"""
        pass
```

---

## 4. Configuration Schema

```yaml
# solver_pipeline_config.yaml

pipeline:
  run_name: "creative_smoke_test_v1"
  seed: 42
  device: "cuda"

checkpoint:
  path: "/storage/models/qwen3-4b-creative-smoke_challenger_v1/"
  format: "auto"  # or explicitly specify
  validate: true

prompt_generation:
  generator_module: "@question_generate/one_shot_creative_question_generate.py"
  num_prompts: 16
  temperature: 0.7
  top_k: 50

solver_sampling:
  num_samples_per_prompt: 1
  max_length: 512
  temperature: 0.8
  top_p: 0.95
  batch_size: 4

evaluation:
  batch_size: 8
  # BatchEvalAgent params passed here

reward_assignment:
  method: "normalized_rank"
  total_samples: 16  # G parameter

output:
  directory: "/path/to/grpo_training_data/"
  format: "jsonl"  # or pickle
  include_metadata: true
```

---

## 5. Main Pipeline Orchestrator

```python
class SolverPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._initialize_components()
    
    def _initialize_components(self):
        """Wire up all components"""
        self.checkpoint_loader = CheckpointLoader()
        self.prompt_generator = PromptGenerator(**self.config.prompt_generation)
        self.solver_sampler = SolverSampler(**self.config.solver_sampling)
        self.evaluator = EvaluationPipeline(batch_eval_agent)
        self.reward_assigner = RewardAssigner(
            total_samples=self.config.reward_assignment.total_samples
        )
    
    def run(self) -> SolverPipelineOutput:
        """Execute full pipeline"""
        
        # Step 1: Load checkpoint
        logger.info(f"Loading checkpoint: {self.config.checkpoint.path}")
        model, metadata = self.checkpoint_loader.load(
            self.config.checkpoint.path
        )
        
        # Step 2: Generate prompts
        logger.info(f"Generating {self.config.prompt_generation.num_prompts} prompts")
        prompts = self.prompt_generator.generate_batch()
        
        # Step 3: Sample outputs
        logger.info(f"Sampling outputs from solver")
        sample_dict = self.solver_sampler.sample_batch(
            prompts=prompts,
            num_samples_per_prompt=self.config.solver_sampling.num_samples_per_prompt
        )
        
        # Step 4: Flatten prompt-sample pairs
        prompt_sample_pairs = self._flatten_samples(prompts, sample_dict)
        
        # Step 5: Evaluate
        logger.info(f"Evaluating {len(prompt_sample_pairs)} samples")
        eval_scores = self.evaluator.evaluate_samples(prompt_sample_pairs)
        
        # Step 6: Assign rewards
        logger.info(f"Computing normalized rank rewards")
        rewards = self.reward_assigner.compute_normalized_rank_rewards(eval_scores)
        
        # Step 7: Assemble output
        output = self._assemble_output(
            prompts, sample_dict, eval_scores, rewards, metadata
        )
        
        # Step 8: Validate and save
        is_valid, errors = output.validate()
        if not is_valid:
            raise PipelineValidationError(errors)
        
        output.save(self.config.output.directory)
        
        return output
    
    def _flatten_samples(self, prompts, sample_dict) -> List[Tuple[str, str]]:
        """Convert {pid: [outputs]} to [(prompt, output), ...]"""
        pass
    
    def _assemble_output(self, ...) -> SolverPipelineOutput:
        """Combine all components into output structure"""
        pass
```

---

## 6. Error Handling and Validation

### 6.1 Validation Checkpoints

```python
class PipelineValidator:
    @staticmethod
    def validate_checkpoint(checkpoint_path: str) -> bool:
        """Verify checkpoint exists and is valid"""
        pass
    
    @staticmethod
    def validate_prompts(prompts: List[str]) -> bool:
        """Check prompts are non-empty, unique"""
        pass
    
    @staticmethod
    def validate_scores(scores: Dict) -> bool:
        """Verify all scores in [0, 1]"""
        pass
    
    @staticmethod
    def validate_rewards(rewards: Dict, expected_mean: float = 0.5) -> bool:
        """Check mean ≈ 0.5 and range [0, 1]"""
        pass
    
    @staticmethod
    def validate_output(output: SolverPipelineOutput) -> Tuple[bool, List[str]]:
        """Full validation before saving"""
        errors = []
        
        if not output.samples:
            errors.append("No samples generated")
        
        reward_mean = np.mean([s.normalized_rank_reward for s in output.samples])
        if abs(reward_mean - 0.5) > 0.05:
            errors.append(f"Reward mean {reward_mean} deviates from 0.5")
        
        return len(errors) == 0, errors
```

---

## 7. Integration with GRPO Training

The pipeline output feeds directly into GRPO training:

```python
# In GRPO training loop:
pipeline_output = solver_pipeline.run()

# Each sample becomes a training example:
for sample in pipeline_output.samples:
    training_example = {
        "prompt": sample.prompt,
        "output": sample.output,
        "reward": sample.normalized_rank_reward,  # ← Used directly
        "metadata": sample.metadata
    }
    # Feed to GRPO optimizer
```

---

## 8. Directory Structure and File Organization

```
/storage/models/qwen3-4b-creative-smoke_challenger_v1/
├── config.json                    # Model config
├── model.safetensors              # Model weights
└── metadata.json                  # Training metadata

/storage/solver_pipeline_outputs/
├── run_20260525_smoke_test_001/
│   ├── config.yaml                # Pipeline config used
│   ├── samples.jsonl              # All solver samples
│   ├── rewards.json               # Reward assignments
│   ├── statistics.json            # Reward stats
│   └── metadata.json              # Run metadata
```

---

## 9. Next Steps for Implementation

1. **Phase 1 - Core Components:**
   - [ ] Implement CheckpointLoader with format auto-detection
   - [ ] Implement PromptGenerator (wrapper around @question_generate)
   - [ ] Implement SolverSampler with reproducible sampling

2. **Phase 2 - Evaluation & Rewards:**
   - [ ] Implement EvaluationPipeline (integration with BatchEvalAgent)
   - [ ] Implement RewardAssigner with validation
   - [ ] Add comprehensive logging

3. **Phase 3 - Orchestration:**
   - [ ] Implement SolverPipeline orchestrator
   - [ ] Add configuration system
   - [ ] Implement full validation suite

4. **Phase 4 - Testing:**
   - [ ] Smoke test with 8-16 samples
   - [ ] Validate reward statistics (mean ≈ 0.5)
   - [ ] Test checkpoint loading flexibility
   - [ ] Integration test with GRPO

---

## 10. Key Design Principles

| Principle | Implementation |
|-----------|-----------------|
| **Modularity** | Each component independently testable, clear interfaces |
| **Traceability** | Every sample records full lineage (checkpoint, prompt, model, timestamp) |
| **Reproducibility** | Configurable seeds, exact config saved with output |
| **Flexibility** | Support multiple checkpoint formats, sampling strategies, evaluation methods |
| **Efficiency** | Batching for evaluation, efficient ranking algorithm |
| **Robustness** | Multi-level validation, clear error messages |

---

## 11. Appendix: Reward Score Proof-of-Concept

```python
# Quick validation that normalized rank rewards work as expected

def verify_reward_properties():
    import numpy as np
    
    G = 16  # Total samples
    ranks = np.arange(1, G + 1)  # [1, 2, ..., 16]
    
    # Compute normalized rewards
    rewards = (G - ranks) / (G - 1)
    
    # Verify properties
    assert np.isclose(np.mean(rewards), 0.5), f"Mean: {np.mean(rewards)}"
    assert np.all(rewards >= 0) and np.all(rewards <= 1), "Range [0, 1]"
    assert rewards[0] == 1.0, "Best rank → reward 1.0"
    assert rewards[-1] == 0.0, "Worst rank → reward 0.0"
    
    print(f"✓ Rewards: {rewards}")
    print(f"✓ Mean: {np.mean(rewards)}")
    print(f"✓ Std: {np.std(rewards)}")
    
    return True
```

