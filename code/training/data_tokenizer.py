import argparse
import ast
from functools import partial
from typing import Dict
from datasets import load_dataset
from transformers import AutoTokenizer

QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()

SYSTEM_PROMPT = "You are a helpful quantum circuit design assistant. You first thinks about the reasoning process in the mind and then provides the user with the optimal answer."

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text

def generate_problem_specific_text(problem: str, attributes: Dict) -> str:
    attributes = ast.literal_eval(attributes)
    if problem == "community_detection":
        return f"with {attributes['communities_size']} sized communities and {attributes['number_of_communities']} communities"
    elif problem == "connected_components":
        return f"for node {attributes['node']}"
    elif problem == "graph_coloring":
        return f"with {attributes['number_of_colors']} colors"
    elif problem == "graph_isomorphism":
        return f"with {attributes['number_of_colors']} colors"
    return ""

def process_graph_example(example: Dict) -> Dict:
    n_qubits = example["number_of_qubits"]
    n_layers = example["number_of_layers"]
    graph = example["graph"]
    circuit_with_params = example["circuit_with_params"]
    circuit_with_symbols = example["circuit_with_symbols"]
    optimization_type = example["optimization_type"]
    problem_type = example["problem_type"]
    problem_specific_text = ""

    if example["problem_specific_attributes"]:
        problem_specific_text = generate_problem_specific_text(problem_type, example["problem_specific_attributes"])

    question = (
        f"Your task is to generate a quantum circuit in QASM 3.0 with {n_qubits} qubits and {n_layers} "
        f" layers with optimal parameters that solves the {problem_type} {problem_specific_text} for "
        f"the following graph: {graph}. Ensure that the final answer is correct and in valid QASM 3.0 code with optimal parameters for the given problem."
    )
    polynom_question = (
        f"Your task is to generate a quantum circuit in QASM 3.0 with {n_qubits} qubits and {n_layers} "
        " layers with optimal parameters that solves the problem {SOME_PROBLEM_DESCRIPTION} using {optimization_type}."
        ". Then ensure that the final answer is correct and in valid QASM 3.0 code."
    )

    # Cost hamiltonian standard?? Qiskit or Pennylane?
    improved_question = (
        f"Your task is to generate a quantum circuit in QASM 3.0 with {n_qubits} qubits and {n_layers} "
        " layers with optimal parameters that solves the problem for the cost hamiltonian {cost_hamiltonian} using {optimization_type}."
        ". Then ensure that the final answer is correct and in valid QASM 3.0 code."
    )

    answer = circuit_with_params
    return dict(
        question=question,
        answer=answer,
        circuit_with_params=circuit_with_params,
        circuit_with_symbols=circuit_with_symbols,
    )

def process_example(example: Dict, tokenizer: AutoTokenizer, mode: str = "sft") -> Dict:
    graph_data = process_graph_example(example)
    question = graph_data["question"]
    answer = graph_data["answer"]
    if "Answer:" not in answer:
        answer = "Answer: " + answer
    if mode == "sft":
        chat_template = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "\n<|im_start|>\n" + answer.strip()},
        ]
        text = tokenizer.apply_chat_template(chat_template, tokenize=False, continue_final_message=True)
        return {"text": text}
    elif mode == "grpo":
        chat_template = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(chat_template, tokenize=False, continue_final_message=True)
        return {"prompt": prompt_text, "target": answer}
    else:
        raise ValueError(f"Unknown mode: {mode}")

def tokenize_examples(download_data_path: str, upload_data_path: str, num_proc: int, mode: str, model: str):
    dataset = load_dataset(download_data_path, download_mode="force_redownload")
    tokenizer = AutoTokenizer.from_pretrained(model)
    process_example_map = partial(process_example, tokenizer=tokenizer, mode=mode)
    if isinstance(dataset, dict) and "train" in dataset:
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                process_example_map,
                num_proc=num_proc,
                desc=f"Tokenizing data for {split} split in {mode} mode"
            )
    else:
        dataset = dataset.map(
            process_example_map,
            num_proc=num_proc,
            desc=f"Tokenizing data in {mode} mode"
        )
    upload_data_path_with_postfix = f"{upload_data_path}_{mode}"
    dataset.push_to_hub(upload_data_path_with_postfix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT or GRPO training.")
    parser.add_argument("--mode", type=str, choices=["sft", "grpo"], required=True, help="SFT or GRPO")
    parser.add_argument("--download_data_path", type=str, default="linuzj/graph-data-quantum", help="Source Dataset Path")
    parser.add_argument("--upload_data_path", type=str, default="linuzj/graph-data-quantum_tokenized", help="Tokenized Dataset Path")
    parser.add_argument("--num_proc", type=int, default=20, help="Processes num.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="(default: Qwen/Qwen2.5-3B-Instruct).")
    args = parser.parse_args()

    tokenize_examples(
        download_data_path=args.download_data_path,
        upload_data_path=args.upload_data_path,
        num_proc=args.num_proc,
        mode=args.mode,
        model=args.model
    )
