# Quantum Circuit Generation

This folder contains tools and scripts for generating quantum circuits using pre-trained language models. The generation system produces QASM 3.0 quantum circuits for various optimization problems including graph-based combinatorial optimization tasks.

## Overview

The generation framework leverages transformer-based language models to automatically generate quantum circuits for solving optimization problems. It supports both single-sample generation and large-scale batch processing with various model configurations and prompt engineering techniques.

## Key Features

- **Multi-Model Support**: Compatible with various language models including Gemma, Qwen, DeepSeek, Llama, and custom quantum-specialized models
- **Few-Shot Learning**: Optional examples in prompts to improve generation quality
- **Batch Processing**: Efficient generation across large datasets
- **SLURM Integration**: Cluster-based distributed generation for scalability
- **Flexible Configuration**: Customizable parameters for different problem types and model settings

## Structure

```text
generation/
├── generate_samples.py          # Main batch generation script
├── run_model.py                # Single sample generation script
├── generate_samples.sh         # SLURM script for batch processing
├── generate_all_samples.sh     # Multi-model SLURM array job
├── run_model.sh                # Single model execution script
├── requirements.txt            # Python dependencies
└── out/                        # Generated circuit outputs (created at runtime)
```

## Usage

### Single Circuit Generation

For quick testing with a single sample:

```bash
python3 run_model.py --uid "test_$(date +%Y%m%d_%H%M%S)" --model_path "model_name"
```

### Batch Generation

Generate circuits for multiple samples from a dataset:

```bash
python3 generate_samples.py \
    --uid "batch_$(date +%Y%m%d_%H%M%S)" \
    --model_path "linuzj/quantum-circuit-qubo-3B" \
    --dataset "linuzj/graph-data-quantum-tokenized_sft" \
    --n_samples 200
```

### Few-Shot Learning

Enable few-shot learning with example circuits in the prompt:

```bash
python3 generate_samples.py \
    --uid "fewshot_$(date +%Y%m%d_%H%M%S)" \
    --model_path "google/gemma-3-4b-it" \
    --dataset "linuzj/graph-data-quantum-tokenized_sft" \
    --n_samples 100 \
    --few_shot_learning
```

### Cluster Processing

For large-scale generation on SLURM clusters:

```bash
# Single model batch generation
sbatch generate_samples.sh

# Multi-model array job
sbatch generate_all_samples.sh
```

## Supported Models

The system has been tested with the following models:

### General Language Models

- **Google Gemma**: `google/gemma-3-4b-it`
- **Qwen Models**: `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Meta Llama**: `meta-llama/Llama-3.2-3B-Instruct`
- **DeepSeek**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **CodeGemma**: `google/codegemma-7b-it`

### Specialized Quantum Models

- **Quantum-QUBO**: `linuzj/quantum-circuit-qubo-3B` (custom fine-tuned model)

## Problem Types

The generation system supports various quantum optimization problems:

### Graph Problems

- **Vertex Cover**: Finding minimum vertex covers in graphs
- **Max-Cut**: Maximum cut partitioning problems
- **Hypergraph Max-Cut**: Generalized max-cut for hypergraphs

### Optimization Algorithms

- **QAOA** (Quantum Approximate Optimization Algorithm): For combinatorial optimization
- **VQE** (Variational Quantum Eigensolver): For energy minimization problems

### Algorithm Categories

- **Hidden Subgroup**: Quantum algorithms for finding hidden structures
- **Logical Operations**: Basic quantum logic and entanglement circuits

## Input Format

The system expects datasets with the following structure:

```json
{
  "signature": "unique_problem_identifier",
  "number_of_qubits": 6,
  "number_of_layers": 2,
  "graph": {"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]]},
  "problem_type": "vertex_cover",
  "problem_specific_attributes": "additional_constraints",
  "optimization_type": "qaoa",
  "circuit_with_params": "OPENQASM 3.0; ...",
  "cost_hamiltonian": "hamiltonian_description",
  "solution": "expected_solution",
  "exact_solution": "optimal_solution"
}
```

## Output Format

Generated circuits are saved as JSON files with comprehensive metadata:

```json
{
  "signature": "problem_signature",
  "model_name": "model_path",
  "sample_index": 0,
  "dataset_metrics": {
    "n_qubits": 6,
    "n_layers": 2,
    "graph": "graph_structure",
    "optimization_type": "qaoa",
    "problem_type": "vertex_cover",
    "problem_specific_attributes": "constraints",
    "optimal_circuit": "reference_circuit",
    "cost_hamiltonian": "hamiltonian",
    "solution": "expected_solution",
    "exact_solution": "optimal_solution"
  },
  "generated_circuit": "OPENQASM 3.0; include \"stdgates.inc\"; ...",
  "generation_time_seconds": 2.45
}
```

## Prompt Engineering

### System Prompt

The system uses a consistent prompt across all models:

```text
You are a helpful quantum circuit design assistant. 
Provide a quantum circuit in valid QASM 3.0 code with optimal gate parameters 
so that the output state encodes the solution, ensuring that the measurement 
outcomes have a high probability of reflecting the correct answer.
```

### Few-Shot Examples

When enabled, the system includes 4 example circuits:

1. **Deutsch Algorithm**: 2-qubit hidden subgroup problem
2. **Entangling Swap**: 2-qubit logical operations
3. **Vertex Cover**: 10-qubit VQE optimization
4. **Hypergraph Max-Cut**: 7-qubit QAOA optimization

### Task-Specific Prompts

Each generation includes problem-specific details:

- Number of qubits and layers
- Graph structure (nodes and edges)
- Problem type and optimization algorithm
- Additional constraints or attributes

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:

- **PyTorch**: Deep learning framework with CUDA/MPS support
- **Transformers**: Hugging Face model loading and inference
- **Datasets**: Hugging Face dataset loading utilities
- **TRL**: Transformer Reinforcement Learning tools
- **Accelerate**: Distributed training and inference
- **WandB**: Experiment tracking and logging

## Hardware Requirements

### GPU Support

- **CUDA**: NVIDIA GPU acceleration (preferred)
- **MPS**: Apple Silicon GPU acceleration (macOS)
- **Memory**: Minimum 15GB GPU memory for 3B+ parameter models

### Cluster Configuration

SLURM job parameters:

- **Time Limit**: 4-36 hours depending on batch size
- **GPU**: 1x H200 or A100 (80GB recommended)
- **CPU**: 2-3 cores per job
- **Memory**: 15-20GB system RAM

## Configuration Options

### Generation Parameters

Key parameters in `generate_samples.py`:

- `max_length`: Maximum generation length (default: 32000 tokens)
- `random_seed`: Reproducibility seed (default: 112)
- `n_samples`: Number of samples to process (default: dataset size)

### Model-Specific Settings

- **Gemma Models**: Use `Gemma3ForCausalLM` with special chat formatting
- **Other Models**: Standard `AutoModelForCausalLM` with generic chat templates
- **Precision**: BFloat16 for memory efficiency

## Performance Optimization

### Memory Management

- Models loaded with `device_map="auto"` for automatic distribution
- BFloat16 precision reduces memory usage by ~50%
- Gradient computation disabled during inference

### Batch Processing

- Single sample processing to avoid memory issues
- Progress tracking with timing information
- Automatic output parsing and validation

## Output Management

### File Naming Convention

```text
quantum_circuits_output_{timestamp}_{model_name}[_few_shot].json
```

Examples:

- `quantum_circuits_output_20250703_143012_quantum-circuit-qubo-3B.json`
- `quantum_circuits_output_20250703_143012_gemma-3-4b-it_few_shot.json`

### Output Directory

Generated files are saved to the `out/` directory (created automatically if it doesn't exist).

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**: Reduce batch size or use smaller models
2. **Model Loading Failures**: Ensure correct model path and access permissions
3. **Dataset Loading Issues**: Verify dataset name and internet connectivity
4. **QASM Parsing Errors**: Check generated circuit syntax in outputs

### Debug Mode

Add verbose logging by modifying the generation scripts:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Resource Monitoring

Monitor GPU usage during generation:

```bash
nvidia-smi -l 1  # NVIDIA GPUs
watch -n 1 'ps aux | grep python'  # CPU/Memory usage
```

## Best Practices

### Model Selection

- Use quantum-specialized models for better circuit quality
- Start with smaller models for testing, scale up for production
- Consider few-shot learning for improved performance

### Dataset Preparation

- Ensure consistent graph representations
- Validate problem parameters before generation
- Use appropriate train/test splits

### Cluster Usage

- Monitor job queues and resource utilization
- Use array jobs for parameter sweeps
- Save intermediate results for long-running jobs

## Contributing

When adding new models or features:

1. Update model loading logic in `generate_samples.py`
2. Test with small sample sizes first
3. Add appropriate chat template formatting
4. Update this README with new capabilities
5. Ensure compatibility with existing evaluation pipeline