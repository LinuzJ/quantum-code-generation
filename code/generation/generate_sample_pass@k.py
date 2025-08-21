#!/usr/bin/env python3
import os
import re
import json
import time
import argparse
import random
from typing import List

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ----------------------------- Constants -----------------------------
SYSTEM_PROMPT = (
    "You are a helpful quantum circuit design assistant. "
    "Provide a quantum circuit in valid QASM 3.0 code with optimal gate parameters so that the "
    "output state encodes the solution, ensuring that the measurement outcomes have a high "
    "probability of reflecting the correct answer."
)

_QASM_START = re.compile(r"OPENQASM\s*3\.0\s*;", re.IGNORECASE)


# ----------------------------- Prompting -----------------------------
def build_user_prompt(sample: dict, few_shot_learning: bool = False) -> str:
    """Create the user content (no role tags)."""
    n_qubits = sample.get("number_of_qubits")
    n_layers = sample.get("number_of_layers")
    graph = sample.get("graph")
    problem_type = sample.get("problem_type")
    attrs = sample.get("problem_specific_attributes")

    few_shot = ""
    if few_shot_learning:
        # Keep examples minimal & valid; avoid "..." that models might copy.
        few_shot = (
            "Example (minimal valid QASM 3.0):\n"
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nbit[2] c;\n'
            "h q[0];\ncx q[0], q[1];\nmeasure q -> c;\n\n"
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
    """Use the model's native chat template for clean prompts."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


# ----------------------------- Post-processing -----------------------------
def extract_qasm(text: str) -> str:
    """
    Extract QASM starting at 'OPENQASM 3.0;' and trim common artifacts
    like markdown fences or trailing role tags.
    """
    if not text:
        return ""

    # Strip code fences if present
    txt = text.strip()
    txt = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)

    m = _QASM_START.search(txt)
    if not m:
        return txt

    qasm = txt[m.start():].strip()

    # If the model continued with another role tag, cut there.
    role_cut = re.search(r"\n<\|\w+\|>", qasm)
    if role_cut:
        qasm = qasm[:role_cut.start()].strip()

    return qasm


# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="vLLM QASM generator (k candidates per prompt)")
    parser.add_argument("--uid", type=str, required=True, help="Unique output ID")
    parser.add_argument("--model_path", type=str, required=True, help="vLLM model path or HF repo")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional tokenizer path/repo")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset path (e.g., user/ds)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument("--n_samples", type=int, default=None, help="Optional subsample size")
    parser.add_argument("--few_shot_learning", action="store_true", help="Enable few-shot examples")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling")
    parser.add_argument("--n_per_prompt", type=int, default=1, help="Number of candidates per prompt")
    # Optional vLLM engine knobs (helpful on HPC)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_mem_util", type=float, default=None)
    args = parser.parse_args()

    # Seed for reproducibility of sampling order (note: diversity still depends on temperature/top_p)
    random_seed = 112
    random.seed(random_seed)

    # Load dataset
    dataset = load_dataset(args.dataset, split=args.split)
    original_size = len(dataset)
    if args.n_samples and args.n_samples < original_size:
        indices = random.sample(range(original_size), args.n_samples)
        dataset = dataset.select(indices)
        print(f"Selected {args.n_samples} samples from {original_size}.")
    else:
        print(f"Using full dataset with {original_size} samples.")

    # Tokenizer (for chat template + stop ids)
    tok_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # Build prompts via chat template
    prompts: List[str] = []
    for sample in dataset:
        user_prompt = build_user_prompt(sample, few_shot_learning=args.few_shot_learning)
        prompts.append(wrap_with_chat_template(tokenizer, user_prompt))

    # Initialize vLLM
    llm_kwargs = dict(model=args.model_path, dtype="bfloat16")
    if args.tensor_parallel_size:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_mem_util:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_mem_util
    llm = LLM(**llm_kwargs)

    # Stop token ids (handle None or list)
    stop_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            stop_ids.extend(tokenizer.eos_token_id)
        else:
            stop_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n_per_prompt,               # <-- request k candidates
        stop=["```"],                      # try to prevent fenced blocks
        stop_token_ids=stop_ids,
        seed=random_seed,
    )

    print(f"Running vLLM generation (n_per_prompt={args.n_per_prompt}) ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - t0
    per_prompt_time = total_time / max(1, len(outputs))

    results = []
    for idx, out in enumerate(outputs):
        sample = dataset[idx]

        # Collect all candidates for this prompt
        candidates = []
        for j, cand in enumerate(out.outputs or []):
            full_text = (cand.text or "").strip()
            generated_circuit = extract_qasm(full_text)
            candidates.append(
                {
                    "candidate_index": j,
                    "raw_text": full_text,
                    "generated_circuit": generated_circuit,
                }
            )

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
            "generations": candidates,                # list of k candidates (raw + extracted)
            "n_generations": len(candidates),
            "generation_time_seconds": per_prompt_time,  # avg per prompt
        }

        results.append(result)
        print(f"Sample {idx}: {len(candidates)} candidates")

    # Save
    few_shot_tag = "_few_shot" if args.few_shot_learning else ""
    model_name_out = args.model_path.split("/")[-1]
    os.makedirs("out", exist_ok=True)
    output_file = f"out/quantum_circuits_output_{args.uid}_{model_name_out}{few_shot_tag}_n{args.n_per_prompt}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n✅ Generation complete.")
    print(f"Avg time per prompt: {per_prompt_time:.2f}s  |  Total: {total_time:.2f}s")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
