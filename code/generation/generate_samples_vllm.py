import os
import json
import time
import argparse
import random
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Constants
SYSTEM_PROMPT = (
    "You are a helpful quantum circuit design assistant. "
    "Provide a quantum circuit in valid QASM 3.0 code with optimal gate parameters so that the output state encodes the solution, ensuring that the measurement outcomes have a high probability of reflecting the correct answer."
)

def create_chat_prompt(sample, few_shot_learning=False):
    n_qubits = sample.get("number_of_qubits")
    n_layers = sample.get("number_of_layers")
    graph = sample.get("graph")
    problem_type = sample.get("problem_type")
    attrs = sample.get("problem_specific_attributes")

    prompt = ""

    if few_shot_learning:
        prompt += (
            "Here are some examples of quantum circuits in QASM 3.0 format:\n"
            "[Example 1] Deutsch algorithm...\n"
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[2] c;\nqubit[2] q;\nx q[1];\nh q[0];\n...\n'
            "[Example 2] Swapping gate problem...\n"
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[2] c;\nqubit[2] q;\n...\n'
            "[Example 3] Vertex cover using VQE...\n"
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[10] c;\nqubit[10] q;\n...\n'
            "[Example 4] Hypermaxcut using QAOA...\n"
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[7] c;\nqubit[7] q;\n...\n'
        )

    prompt += (
        f"Your task is to generate a quantum circuit in QASM 3.0 with "
        f"{n_qubits} qubits and {n_layers} layers with optimal parameters that solves the "
        f"{problem_type} {attrs} problem for the following graph: {graph}. "
        "Then ensure that the final answer is correct and in valid QASM 3.0 code. "
        "Only return the full QASM code, nothing else."
    )

    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"


def main():
    parser = argparse.ArgumentParser(description="Run vLLM quantum circuit generation")
    parser.add_argument("--uid", type=str, required=True, help="Unique output ID")
    parser.add_argument("--model_path", type=str, required=True, help="vLLM model path or name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path (HuggingFace)")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--few_shot_learning", action="store_true", help="Enable few-shot prompting")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    # Seed for reproducibility
    random_seed = 112
    random.seed(random_seed)

    # Load dataset
    dataset = load_dataset(args.dataset, split="test")
    original_size = len(dataset)
    if args.n_samples and args.n_samples < original_size:
        indices = random.sample(range(original_size), args.n_samples)
        dataset = dataset.select(indices)
        print(f"Selected {args.n_samples} samples from {original_size}.")
    else:
        print(f"Using full dataset with {original_size} samples.")

    # Load tokenizer (for eos_token, etc.)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Initialize vLLM
    llm = LLM(model=args.model_path, dtype="bfloat16")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        stop=[],
        stop_token_ids=[tokenizer.eos_token_id],
    )

    prompts = []
    for sample in dataset:
        prompts.append(create_chat_prompt(sample, few_shot_learning=args.few_shot_learning))

    print("Running vLLM generation...")

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start

    results = []

    for idx, output in enumerate(outputs):
        sample = dataset[idx]
        full_text = output.outputs[0].text.strip()

        # Try to extract starting from "OPENQASM 3.0;"
        assistant_indicator = "OPENQASM 3.0;"
        loc = full_text.rfind(assistant_indicator)
        generated_circuit = full_text[loc:].strip() if loc != -1 else full_text

        result = {
            "signature": sample.get("signature"),
            "model_name": args.model_path,
            "sample_index": idx,
            "dataset_metrics": {
                "n_qubits": sample.get("number_of_qubits"),
                "n_layers": sample.get("number_of_layers"),
                "graph": sample.get("graph"),
                "optimization_type": sample.get("optimization_type"),
                "problem_type": sample.get("problem_type"),
                "problem_specific_attributes": sample.get("problem_specific_attributes"),
                "optimal_circuit": sample.get("circuit_with_params"),
                "cost_hamiltonian": sample.get("cost_hamiltonian"),
                "solution": sample.get("solution"),
                "exact_solution": sample.get("exact_solution"),
            },
            "generated_circuit": generated_circuit,
            "generation_time_seconds": total_time / len(outputs),
        }

        results.append(result)
        print(f"Sample {idx}: generated in {result['generation_time_seconds']:.2f}s")

    few_shot = "_few_shot" if args.few_shot_learning else ""
    model_name_out = args.model_path.split("/")[-1]
    output_file = f"out/quantum_circuits_output_{args.uid}_{model_name_out}{few_shot}.json"

    os.makedirs("out", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Generation complete. Output saved to: {output_file}")


if __name__ == "__main__":
    main()
