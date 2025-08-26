import os
import re
import json
import time
import argparse
import random
from typing import List, Optional

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You are a helpful quantum circuit design assistant. "
    "Provide a quantum circuit in valid QASM 3.0 code with optimal gate parameters so that the "
    "output state encodes the solution, ensuring that the measurement outcomes have a high "
    "probability of reflecting the correct answer."
)

def build_user_prompt(sample, few_shot_learning: bool = False) -> str:
    n_qubits = sample.get("number_of_qubits")
    n_layers = sample.get("number_of_layers")
    graph = sample.get("graph")
    problem_type = sample.get("problem_type")
    attrs = sample.get("problem_specific_attributes")

    few_shot = ""
    if few_shot_learning:
        # Prefer minimal valid examples; avoid ellipses that bleed into generations.
        few_shot = (
            "Examples of minimal QASM 3.0 circuits:\n"
            "[Example] Bell state preparation\n"
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nmeasure q -> c;\n\n'
        )

    task = (
        f"Your task is to generate a quantum circuit in QASM 3.0 with "
        f"{n_qubits} qubits and {n_layers} layers that solves the "
        f"{problem_type} {attrs} problem for the following graph: {graph}.\n"
        "Return only the full QASM 3.0 program (no explanations, no markdown fences). "
        'Begin with the line: OPENQASM 3.0;'
    )
    return (few_shot + task).strip()

def wrap_with_chat_template(tokenizer, user_prompt: str) -> str:
    # Use the model’s native chat template for clean prompting.
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

_QASM_START = re.compile(r"OPENQASM\s*3\.0\s*;", re.IGNORECASE)

def extract_qasm(text: str) -> str:
    """
    Extract QASM starting at 'OPENQASM 3.0;' and strip code fences / stray role tags.
    """
    # Remove markdown fences if present
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)

    m = _QASM_START.search(text)
    if not m:
        return text.strip()

    qasm = text[m.start():].strip()

    # Stop at next role tag if the model continued the chat
    role_cut = re.search(r"\n<\|\w+\|>", qasm)
    if role_cut:
        qasm = qasm[:role_cut.start()].strip()

    return qasm

def main():
    parser = argparse.ArgumentParser(description="Run vLLM quantum circuit generation")
    parser.add_argument("--uid", type=str, required=True, help="Unique output ID")
    parser.add_argument("--model_path", type=str, required=True, help="vLLM model path or name")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional tokenizer path/name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path (HuggingFace)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--few_shot_learning", action="store_true", help="Enable few-shot prompting")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    # Optional vLLM engine knobs
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_mem_util", type=float, default=None)
    args = parser.parse_args()

    # Seed for reproducibility
    py_seed = 112
    random.seed(py_seed)

    # Load dataset
    dataset = load_dataset(args.dataset, split=args.split)
    original_size = len(dataset)
    if args.n_samples and args.n_samples < original_size:
        indices = random.sample(range(original_size), args.n_samples)
        dataset = dataset.select(indices)
        print(f"Selected {args.n_samples} samples from {original_size}.")
    else:
        print(f"Using full dataset with {original_size} samples.")

    # Tokenizer
    tok_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # Build prompts via chat template
    prompts: List[str] = []
    for sample in dataset:
        user_prompt = build_user_prompt(sample, few_shot_learning=args.few_shot_learning)
        prompts.append(wrap_with_chat_template(tokenizer, user_prompt))

    # vLLM engine init
    llm_kwargs = dict(model=args.model_path, dtype="bfloat16")
    if args.tensor_parallel_size:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_mem_util:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_mem_util

    llm = LLM(**llm_kwargs)

    # Stop IDs/strings
    stop_token_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            stop_token_ids.extend(tokenizer.eos_token_id)
        else:
            stop_token_ids.append(tokenizer.eos_token_id)

    # Some chat templates add special assistant-stop tokens; include them if exposed
    stop_strs: List[str] = []
    # Avoid markdown fences in outputs
    stop_strs.extend(["```"])

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        stop=stop_strs,
        stop_token_ids=stop_token_ids,
        seed=py_seed,
    )

    print("Running vLLM generation...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - t0

    results = []
    per_sample_time = total_time / max(1, len(outputs))

    for idx, output in enumerate(outputs):
        sample = dataset[idx]
        if not output.outputs:
            gen_text = ""
        else:
            gen_text = (output.outputs[0].text or "").strip()

        generated_circuit = extract_qasm(gen_text)

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
            "raw_text": gen_text,  # keep for debugging; remove if large
            "generation_time_seconds": per_sample_time,
        }

        results.append(result)
        print(f"Sample {idx}: generated in {per_sample_time:.2f}s")

    few_shot = "_few_shot" if args.few_shot_learning else ""
    model_name_out = args.model_path.split("/")[-1]
    os.makedirs("out", exist_ok=True)
    output_file = f"out/quantum_circuits_output_{args.uid}_{model_name_out}{few_shot}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "meta": {
                    "dataset": args.dataset,
                    "split": args.split,
                    "n_samples": len(results),
                    "total_time_seconds": total_time,
                    "per_sample_seconds_mean": per_sample_time,
                    "model": args.model_path,
                    "tokenizer": tok_path,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n✅ Generation complete. Output saved to: {output_file}")

if __name__ == "__main__":
    main()