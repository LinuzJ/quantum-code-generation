# Quantum Circuit Data Generation

This folder contains tools and scripts for generating training datasets of quantum circuits that solve various graph optimization problems. The system automatically creates quantum circuits using QAOA and VQE algorithms with optimal parameters for different combinatorial optimization tasks.

## Overview

The data generation framework produces large-scale datasets of quantum optimization problems paired with their optimal quantum circuit solutions. It supports multiple graph-based optimization problems, various quantum algorithms, and different ansatz templates, making it suitable for training quantum circuit generation models.

## Key Features

- **Multi-Problem Support**: 13 different graph optimization problems
- **Quantum Algorithms**: QAOA, VQE, and Adaptive VQE implementations
- **Ansatz Variety**: 15+ different ansatz templates for VQE
- **Parallel Processing**: Multi-core and GPU acceleration support
- **Automatic Validation**: Circuit compilation and solution verification
- **Scalable Generation**: SLURM cluster integration for large datasets
- **Data Analysis**: Built-in visualization and summarization tools

## Structure

```text
data_generation/
├── src/                              # Core generation modules
│   ├── main.py                      # Main entry point
│   ├── data_generator.py            # Core data generation logic
│   ├── data_classes.py              # Data structures and enums
│   ├── binary_optimization_problem.py # QUBO problem formulations
│   ├── solver.py                    # Quantum optimization solvers
│   ├── ansatz.py                    # Quantum circuit ansatz definitions
│   ├── utils.py                     # Utility functions
│   ├── algorithms/                  # Problem-specific implementations
│   │   ├── vertex_cover/           # Vertex cover problem
│   │   ├── hypermaxcut/           # Hypergraph max-cut
│   │   ├── kcliques/              # k-clique problems
│   │   ├── matching/              # Graph matching
│   │   ├── graph_coloring/        # Graph coloring
│   │   ├── community_detection/   # Community detection
│   │   ├── connected_components/  # Connected components
│   │   ├── edge_cover/           # Edge cover
│   │   ├── graph_isomorphism/    # Graph isomorphism
│   │   ├── hamiltonian_path/     # Hamiltonian path
│   │   ├── max_flow/             # Maximum flow
│   │   ├── min_cut/              # Minimum cut
│   │   └── steiner_tree/         # Steiner tree
│   └── optimization/               # Quantum optimization methods
│       ├── ansatz.py              # Base ansatz classes
│       └── ansatzes/              # Specific ansatz implementations
├── training_data_gen.sh            # SLURM batch generation script
├── generate_hypergraphs.sh         # Hypergraph generation script
├── summarize_data.py               # Dataset analysis and visualization
├── analyze_circuit_sizes.py        # Circuit complexity analysis
├── sample_and_delete_files.py      # Dataset sampling utilities
├── cancel.sh                       # Job cancellation script
└── requirements.txt                # Python dependencies
```

## Supported Problems

### Graph Optimization Problems

1. **Vertex Cover**: Finding minimum vertex sets covering all edges
2. **Edge Cover**: Finding minimum edge sets covering all vertices
3. **Max-Cut/HyperMaxCut**: Partitioning graphs/hypergraphs for maximum cut
4. **K-Clique**: Finding cliques of size k in graphs
5. **Graph Coloring**: Coloring graphs with minimum colors
6. **Maximum Matching**: Finding maximum cardinality matchings
7. **Community Detection**: Identifying community structures
8. **Connected Components**: Finding connected subgraphs
9. **Graph Isomorphism**: Determining graph structural equivalence
10. **Hamiltonian Path**: Finding paths visiting each vertex once
11. **Maximum Flow**: Computing maximum flow between nodes
12. **Minimum Cut**: Finding minimum cuts separating nodes
13. **Steiner Tree**: Finding minimum trees connecting terminal nodes

### Quantum Algorithms

- **QAOA** (Quantum Approximate Optimization Algorithm): For combinatorial optimization
- **VQE** (Variational Quantum Eigensolver): For ground state problems
- **Adaptive VQE**: Dynamic circuit construction with gradient-based growth

### Ansatz Templates (VQE)

15+ predefined ansatz templates including:

- Hardware-efficient ansätze
- Problem-inspired ansätze
- Chemically-motivated ansätze
- Custom parametrized circuits

## Usage

### Single Problem Generation

```bash
# Generate QAOA circuits for vertex cover
python -m src.main \
    --problem "vertex_cover" \
    --layers 2 \
    --output_path "./output"

# Generate VQE circuits with specific ansatz
python -m src.main \
    --problem "hypermaxcut" \
    --layers 3 \
    --ansatz_template 5 \
    --output_path "./output" \
    --vqe
```

### Batch Generation

```bash
# Submit SLURM array job for multiple problems
sbatch training_data_gen.sh
```

This generates circuits for:

- Multiple problems (vertex_cover, edge_cover, etc.)
- Different layer counts (1, 2, 3, 4)
- Various ansatz templates (1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18)
- Both QAOA and VQE approaches

### Hypergraph Generation

```bash
# Generate hypergraph-specific datasets
./generate_hypergraphs.sh
```

## Configuration Options

### Problem Selection

```python
class OptimizationProblemType(str, Enum):
    CONNECTED_COMPONENTS = "connected_components"
    COMMUNITY_DETECTION = "community_detection"
    K_CLIQUE = "kclique"
    HYPERMAXCUT = "hypermaxcut"
    GRAPH_ISOMORPHISM = "graph_isomorphism"
    GRAPH_COLORING = "graph_coloring"
    HAMILTONIAN_PATH = "hamiltonian_path"
    MATCHING = "matching"
    MAX_FLOW = "max_flow"
    MIN_CUT = "min_cut"
    STEINER_TREE = "steiner_tree"
    VERTEX_COVER = "vertex_cover"
    EDGE_COVER = "edge_cover"
```

### Algorithm Types

```python
class OptimizationType(str, Enum):
    VQE = "vqe"
    ADAPTIVE_VQE = "adaptive_vqe"
    QAOA = "qaoa"
```

### Hardware Settings

```bash
# CPU-only execution
export JAX_PLATFORM_NAME="cpu"
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS=4

# GPU acceleration (automatic detection)
# GPU devices are automatically used if available
```

## Output Format

### Generated Data Structure

```json
{
  "signature": "unique_problem_identifier",
  "problem_type": "vertex_cover",
  "optimization_type": "qaoa",
  "graph": {
    "nodes": [0, 1, 2, 3],
    "edges": [[0, 1], [1, 2], [2, 3]]
  },
  "number_of_qubits": 4,
  "number_of_layers": 2,
  "circuit_with_params": "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n...",
  "circuit_with_symbols": "symbolic_circuit_representation",
  "cost_hamiltonian": "hamiltonian_description",
  "solution": {
    "states": [0, 1, 1, 0],
    "expectation_value": -2.5,
    "bitstrings": ["0110"],
    "probabilities": [0.95, 0.05],
    "optimization_time": 12.34
  },
  "exact_solution": {
    "smallest_eigenvalues": -3.0,
    "number_of_smallest_eigenvalues": 1,
    "first_excited_energy": -2.8,
    "smallest_bitstrings": ["0110"]
  },
  "problem_specific_attributes": {
    "vertex_cover": [1, 2]
  },
  "adaptive_process": {
    "circuits": ["layer1_circuit", "layer2_circuit"],
    "gradients": [0.1, 0.05]
  }
}
```

### File Naming Convention

```text
{problem_type}_OptimizationType.{algorithm}_{layers}layers_ansatz{template_id}_{timestamp}.json
```

Examples:

- `vertex_cover_OptimizationType.qaoa_2layers_ansatz1_20250703_143012.json`
- `hypermaxcut_OptimizationType.vqe_3layers_ansatz5_20250703_143012.json`

## Hardware Requirements

### Minimum Requirements

- **CPU**: 8+ cores for parallel processing
- **RAM**: 32GB for large graph problems
- **Storage**: 100GB+ for dataset storage

### Recommended Setup

- **CPU**: 32+ cores (HPC cluster)
- **RAM**: 64-128GB
- **GPU**: CUDA-compatible for JAX acceleration
- **Storage**: 1TB+ NVMe SSD

### Cluster Configuration

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=75GB
#SBATCH --time=10:00:00
#SBATCH --array=0-119
```

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- **JAX**: High-performance computing with GPU support
- **PennyLane**: Quantum machine learning framework
- **Qiskit**: Quantum circuit manipulation and QASM generation
- **NetworkX**: Graph algorithms and data structures
- **DIMOD**: Binary optimization problem formulations
- **NumPy/SciPy**: Numerical computations
- **Matplotlib**: Data visualization

## Data Analysis

### Dataset Summarization

```bash
# Generate visualization of dataset statistics
python summarize_data.py --input_dir "./output" --output_file "summary.png"
```

Features:
- Problem type distribution
- Algorithm type breakdown
- Circuit size analysis
- Generation time statistics

### Circuit Size Analysis

```bash
# Analyze circuit complexity metrics
python analyze_circuit_sizes.py --data_dir "./output"
```

Metrics:
- Gate count distributions
- Qubit usage patterns
- Circuit depth analysis
- Parameter count statistics

### Dataset Sampling

```bash
# Sample subset of generated data
python sample_and_delete_files.py --input_dir "./output" --sample_rate 0.1
```

## Performance Optimization

### Parallel Processing
- Multi-core CPU utilization for graph generation
- Process-based parallelism for independent problems
- JAX JIT compilation for quantum circuit optimization

### Memory Management
- Streaming data processing for large datasets
- Automatic cleanup of intermediate results
- Configurable batch sizes for memory-constrained environments

### GPU Acceleration
- JAX GPU backend for quantum optimization
- Automatic device detection and utilization
- CUDA 12 support for latest hardware

## Quality Assurance

### Validation Pipeline
- Automatic QASM circuit compilation testing
- Solution verification against classical solvers
- Energy expectation value validation
- Statistical analysis of optimization convergence

### Error Handling
- Robust exception handling for problematic graphs
- Automatic retry mechanisms for optimization failures
- Comprehensive logging for debugging

## Best Practices

### Problem Selection
- Start with smaller problems (vertex cover, edge cover) for testing
- Use appropriate layer counts: 1-2 for QAOA, 2-4 for VQE
- Consider problem complexity vs. available quantum resources

### Algorithm Choice
- **QAOA**: Better for combinatorial optimization problems
- **VQE**: More suitable for ground state problems
- **Adaptive VQE**: Use when circuit structure is unknown

### Resource Management
- Monitor memory usage during large batch jobs
- Use appropriate SLURM resource requests
- Clean up intermediate files regularly

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or enable streaming mode
2. **Optimization Failures**: Adjust convergence criteria or try different ansätze
3. **QASM Generation Errors**: Check circuit parameter bounds and gate definitions
4. **Graph Generation Issues**: Validate input graph structures and connectivity

### Debug Mode

```bash
# Enable detailed logging
export PYTHONPATH="${PYTHONPATH}:."
python -u -m src.main --problem "vertex_cover" --layers 1 --output_path "./debug"
```

### Performance Monitoring

```bash
# Monitor resource usage
htop  # CPU and memory
nvidia-smi -l 1  # GPU utilization
df -h  # Disk space
```

## Contributing

When adding new optimization problems:

1. **Create Problem Class**: Add to `src/algorithms/new_problem/`
2. **Implement QUBO**: Define binary optimization formulation
3. **Add to Factory**: Register in `src/algorithms/factory.py`
4. **Update Enums**: Add to `OptimizationProblemType`
5. **Add Tests**: Validate problem correctness
6. **Update Documentation**: Extend this README

### Development Workflow

```bash
# Set up development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Generate small test dataset
python -m src.main --problem "vertex_cover" --layers 1 --output_path "./test_output"
```