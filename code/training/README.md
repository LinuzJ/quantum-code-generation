# Quantum Circuit Model Training

This folder contains tools and scripts for training language models to generate quantum circuits. The training system supports both supervised fine-tuning (SFT) and reinforcement learning approaches including Group Relative Policy Optimization (GRPO) for quantum circuit generation tasks.

## Overview

The training framework fine-tunes pre-trained language models on quantum circuit generation tasks, teaching them to produce valid QASM 3.0 code for solving optimization problems. It supports distributed training, advanced optimization techniques, and quantum-specific reward functions.

## Key Features

- **Supervised Fine-Tuning (SFT)**: Standard instruction-following fine-tuning on quantum circuit datasets
- **Group Relative Policy Optimization (GRPO)**: Reinforcement learning with quantum-specific reward functions
- **Distributed Training**: Multi-GPU support with FSDP and DeepSpeed acceleration
- **Quantum Validation**: Real-time circuit compilation and execution validation during training
- **Data Pipeline**: Comprehensive data preprocessing and tokenization tools
- **Flexible Configuration**: YAML-based configuration for easy experiment management

## Structure

```text
training/
├── sft.py                      # Supervised fine-tuning script
├── grpo.py                     # Group Relative Policy Optimization training
├── grpo_reward_functions.py    # Quantum-specific reward functions
├── data_tokenizer.py          # Data preprocessing and tokenization
├── upload_data.py             # Dataset uploading utilities
├── merge_json.py              # JSON dataset merging tools
├── sft_distributed.sh         # Multi-GPU SFT training script
├── sft_single.sh              # Single GPU SFT training script
├── grpo_distributed.sh        # Multi-GPU GRPO training script
├── ds_config.yaml             # DeepSpeed configuration
├── fsdp_config.yaml           # FSDP configuration
├── fsdp_config.json           # FSDP JSON configuration
└── requirements.txt           # Python dependencies
```

## Training Methods

### Supervised Fine-Tuning (SFT)

Standard instruction-following training on quantum circuit datasets:

```bash
# Single GPU training
python sft.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --train_file_path "linuzj/graph-data-quantum-tokenized_sft" \
    --output_dir "quantum-circuit-model" \
    --num_train_epochs 5

# Multi-GPU distributed training
sbatch sft_distributed.sh
```

**Key Features:**

- Instruction template-based training with proper loss masking
- Only computes loss on assistant responses (not user prompts)
- Flash Attention 2 for memory efficiency
- BFloat16 mixed precision training
- Automatic model checkpointing and resumption

### Group Relative Policy Optimization (GRPO)

Reinforcement learning approach with quantum-specific reward functions:

```bash
# GRPO training with multiple reward functions
python grpo.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --train_file_path "linuzj/graph-data-quantum-basic-optimizer_tokenized_grpo" \
    --output_dir "quantum-circuit-grpo-model"
```

**Reward Functions:**

- **Format Reward**: Validates proper `<think>` and `<answer>` tag structure
- **Compilation Reward**: Ensures generated circuits compile successfully
- **Probability Distribution Reward**: Measures similarity to optimal circuit states
- **Length Reward**: Encourages appropriate circuit complexity

## Supported Models

### Base Models

- **Qwen Series**: `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
- **Gemma Models**: `google/gemma-3-4b-it`
- **Llama Models**: `meta-llama/Llama-3.2-3B-Instruct`
- **Custom Models**: Any transformer-based causal language model

### Output Models

- **Quantum-QUBO-3B**: `linuzj/quantum-circuit-qubo-3B`
- **Quantum-QUBO-7B**: `linuzj/quantum-circuit-qubo-7B`

## Data Format

### SFT Dataset Format

```json
{
  "text": "<|im_start|>system\nYou are a helpful quantum circuit design assistant...<|im_end|>\n<|im_start|>user\nGenerate a quantum circuit...<|im_end|>\n<|im_start|>assistant\nOPENQASM 3.0;\ninclude \"stdgates.inc\";\n...<|im_end|>"
}
```

### GRPO Dataset Format

```json
{
  "text": "<|im_start|>user\nGenerate a quantum circuit...<|im_end|>",
  "solution": "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n..."
}
```

### Raw Data Structure

```json
{
  "signature": "unique_problem_id",
  "problem_type": "vertex_cover",
  "optimization_type": "qaoa",
  "graph": {"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]]},
  "number_of_qubits": 6,
  "number_of_layers": 2,
  "circuit_with_params": "OPENQASM 3.0; ...",
  "cost_hamiltonian": "hamiltonian_description",
  "solution": {"optimal_bitstring": "101010"},
  "exact_solution": {"energy": -2.5}
}
```

## Configuration

### Training Hyperparameters

**SFT Configuration:**
- **Learning Rate**: 2e-5 (default TRL setting)
- **Batch Size**: 1 per device (memory optimized)
- **Gradient Accumulation**: 1-8 steps depending on GPU memory
- **Sequence Length**: 10,000-16,384 tokens
- **Epochs**: 5-15 depending on dataset size

**GRPO Configuration:**
- **Learning Rate**: 1e-6 (lower for RL stability)
- **PPO Epochs**: 4 per batch
- **Mini Batch Size**: 1-2 samples
- **KL Penalty**: 0.1 for policy constraint

### Memory Optimization

**FSDP (Fully Sharded Data Parallel):**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
```

**DeepSpeed ZeRO:**
```yaml
zero_stage: 1  # Parameter sharding
train_micro_batch_size_per_gpu: 2
gradient_clipping: 1.0
bf16: true
```

## Data Processing

### Dataset Preparation

```bash
# Process raw JSON data into tokenized format
python data_tokenizer.py \
    --input_file "raw_quantum_data.json" \
    --output_dataset "processed_dataset" \
    --tokenizer_name "Qwen/Qwen2.5-3B-Instruct"
```

**Features:**
- Multiple prompt variations for data augmentation
- Proper instruction template formatting
- Reasoning template support (`<think>` and `<answer>` tags)
- Automatic data validation and cleaning

### Data Upload

```bash
# Upload processed dataset to Hugging Face Hub
python upload_data.py \
    --dataset_path "processed_dataset" \
    --repo_name "username/quantum-circuit-dataset"
```

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 24GB for 3B models, 80GB for 7B models
- **System RAM**: 64GB recommended
- **Storage**: 100GB+ for model checkpoints and datasets

### Recommended Setup
- **Multi-GPU**: 4-8x H100/H200 80GB GPUs
- **Network**: InfiniBand for multi-node training
- **Storage**: NVMe SSD for fast data loading

## Cluster Training

### SLURM Configuration

**Single Node Multi-GPU:**
```bash
#SBATCH --nodes=1
#SBATCH --gpus=6
#SBATCH --mem=400GB
#SBATCH --time=04:00:00
```

**Multi-Node Training:**
```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=800GB
```

### Environment Setup

```bash
module load scicomp-python-env/2024-01
module load scicomp-llm-env
source .venv/bin/activate
export WANDB_API_KEY=$(cat .wandb_api_key)
```

## Reward Functions (GRPO)

### Format Validation
```python
def format_reward(completions, **kwargs):
    """Validates proper <think> and <answer> tag structure"""
    pattern = r"(?s)^<think>.*?</think>\s*<answer>(.*?)</answer>$"
    matches = [re.match(pattern, content) for content in completions]
    return [1.0 if match else 0.0 for match in matches]
```

### Circuit Compilation
```python
def circuit_compile_reward(completions, **kwargs):
    """Ensures generated circuits compile successfully"""
    rewards = []
    for completion in completions:
        try:
            circuit = parse_qasm_circuit_from_str(completion)
            rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards
```

### Probability Distribution Similarity
```python
def probability_distrubution_reward(completions, **kwargs):
    """Measures state similarity to optimal circuits"""
    # Computes KL divergence between generated and optimal states
    # Returns 1/(1 + relative_entropy) for numerical stability
```

## Monitoring and Logging

### Weights & Biases Integration

```python
# Automatic experiment tracking
os.environ["WANDB_PROJECT"] = "quantum-circuit-generation"
args.report_to = ["wandb"]
```

**Tracked Metrics:**
- Training/validation loss
- Reward function values (GRPO)
- Circuit compilation success rates
- Generation quality metrics
- Hardware utilization

### Checkpointing

```bash
# Automatic checkpoint saving
--save_strategy="steps"
--save_steps=1000
--push_to_hub=True
--hub_strategy=all_checkpoints
```

## Performance Optimization

### Memory Efficiency
- **Flash Attention 2**: Reduces memory usage for long sequences
- **BFloat16**: Halves memory requirements vs Float32
- **Gradient Checkpointing**: Trades compute for memory
- **Parameter Sharding**: Distributes model across GPUs

### Training Speed
- **Compiled Models**: PyTorch 2.0 compilation for speed
- **Data Loading**: Parallel data processing
- **Mixed Precision**: Faster computation with maintained accuracy

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size or sequence length
   - Enable gradient checkpointing
   - Use parameter sharding (FSDP/DeepSpeed)

2. **Circuit Compilation Failures**
   - Check QASM syntax in training data
   - Validate quantum gate parameters
   - Ensure proper include statements

3. **Training Instability**
   - Lower learning rate for RL training
   - Increase gradient clipping threshold
   - Check reward function scaling

### Debug Mode

```bash
# Enable verbose logging
export TRANSFORMERS_VERBOSITY=debug
export WANDB_MODE=disabled  # Disable logging for debugging
```

### Performance Monitoring

```bash
# GPU utilization
nvidia-smi -l 1

# Memory usage
watch -n 1 'free -h'

# Training progress
tail -f logs/training.log
```

## Best Practices

### Model Selection
- Start with smaller models (3B) for prototyping
- Use quantum-specialized models as base when available
- Consider instruction-tuned models for better prompt following

### Data Quality
- Validate all QASM circuits before training
- Balance dataset across different problem types
- Include diverse graph structures and sizes

### Training Strategy
- Begin with SFT for basic capabilities
- Follow with GRPO for optimization and validation
- Use progressive training: simple → complex problems

### Evaluation
- Monitor compilation rates during training
- Validate circuits on unseen test problems
- Compare against classical optimization baselines

## Contributing

When adding new features:

1. **New Reward Functions**: Add to `grpo_reward_functions.py`
2. **Data Formats**: Update `data_tokenizer.py` processing logic
3. **Model Support**: Extend model loading in training scripts
4. **Configurations**: Add new YAML configs for different setups
5. **Documentation**: Update this README with new capabilities

### Testing

```bash
# Validate data processing
python data_tokenizer.py --test_mode

# Test reward functions
python -c "from grpo_reward_functions import *; test_all_rewards()"

# Dry run training
python sft.py --max_steps 10 --output_dir test_output
```