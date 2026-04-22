import argparse
import ast
import random
import json
from functools import partial
from typing import Dict
from datasets import load_dataset
from transformers import AutoTokenizer

QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()

SYSTEM_PROMPT = (
    "You are a helpful quantum circuit design assistant. "
    "Provide a quantum circuit in valid QASM 3.0 code with optimal gate parameters so that the output state encodes the solution, ensuring that the measurement outcomes have a high probability of reflecting the correct answer."
)

RESONING_ADDITION_SYSTEM_PROMT = (
    "The user asks a question, and the you, the Assistant, solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def generate_prompt_permutations(
    n_qubits, n_layers, problem_type, problem_specific_text, graph
):
    prompts = [
        (
            f"Generate a quantum circuit in QASM 3.0 with {n_qubits} qubits "
            f"and {n_layers} layers that solves the {problem_type} {problem_specific_text} "
            f"for the graph: {graph}. The answer must be correct and in valid QASM 3.0 code "
            f"with optimal parameters."
        ),
        (
            f"Given the graph: {graph}, design a QASM 3.0 quantum circuit using {n_qubits} qubits "
            f"and {n_layers} layers to solve the {problem_type} {problem_specific_text}. "
            f"Output valid QASM 3.0 code with optimal parameters."
        ),
        (
            f"Construct a quantum circuit in QASM 3.0 with {n_qubits} qubits and {n_layers} layers "
            f"to address the {problem_type} {problem_specific_text} for the graph: {graph}. "
            f"Ensure the final output is valid QASM 3.0 code with optimal parameters."
        ),
        (
            f"Create a quantum circuit in QASM 3.0 featuring {n_qubits} qubits and {n_layers} layers "
            f"that resolves the {problem_type} {problem_specific_text} for the graph: {graph}. "
            f"The answer should be in correct QASM 3.0 code with optimal parameters."
        ),
        (
            f"Design a QASM 3.0 quantum circuit with {n_qubits} qubits and {n_layers} layers "
            f"to solve the {problem_type} {problem_specific_text} given the graph: {graph}. "
            f"Provide valid QASM 3.0 code with optimal parameters."
        ),
    ]
    return prompts


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text

    # CONNECTED_COMPONENTS = "connected_components"
    # COMMUNITY_DETECTION = "community_detection"
    # K_CLIQUE = "kclique"
    # HYPERMAXCUT = "hypermaxcut"
    # GRAPH_ISOMORPHISM = "graph_isomorphism"
    # GRAPH_COLORING = "graph_coloring"
    # HAMILTONIAN_PATH = "hamiltonian_path"
    # MATCHING = "matching"
    # MAX_FLOW = "max_flow"
    # MIN_CUT = "min_cut"
    # STEINER_TREE = "steiner_tree"


def generate_problem_specific_text(problem: str, attributes: Dict) -> str:
    if not isinstance(attributes, dict):
        try:
            attributes = ast.literal_eval(attributes)
        except Exception as e:
            raise ValueError(f"Could not parse attributes: {attributes}") from e

    if problem == "community_detection":
        return f"with {attributes['communities_size']} sized communities and {attributes['number_of_communities']} communities"
    elif problem == "connected_components":
        return f"for node {attributes['node']}"
    elif problem == "graph_coloring":
        return f"with {attributes['number_of_colors']} colors"
    elif problem == "graph_isomorphism":
        return f"with the second graph being {attributes['autoisomorphic_graph']}"
    elif problem == "hamiltonian_path":
        start_node, end_node = attributes["start_node"], attributes["end_node"]
        return f"starting from node {start_node} and ending at node {end_node}."
    elif problem == "matching":
        matching = attributes["matching"]

        if matching == "maximum":
            matching = "minimal maximum"
        return f"using {matching} matching to solve the problem"
    elif problem == "kclique":
        return f"with {attributes['k']} sized clique"
    elif problem == "max_flow":
        source, sink = attributes["source"], attributes["sink"]
        return f"from source node {source} to sink node {sink}"
    elif problem == "min_cut":
        source, sink = attributes["source"], attributes["sink"]
        return f"from source node {source} to sink node {sink}"
    elif problem == "steiner_tree":
        return f"with the following terminals: {attributes['terminal_nodes']}"

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
        problem_specific_text = generate_problem_specific_text(
            problem_type, json.loads(example["problem_specific_attributes"])
        )

    prompt = generate_prompt_permutations(
        n_qubits, n_layers, problem_type, problem_specific_text, graph
    )
    prompt = random.choice(prompt)

    answer = circuit_with_params
    return dict(
        question=prompt,
        answer=answer,
    )


def process_example(example: Dict, tokenizer: AutoTokenizer, mode: str = "sft") -> Dict:
    graph_data = process_graph_example(example)
    question = graph_data["question"]
    answer = graph_data["answer"]

    if mode == "sft":
        chat_template = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer.strip()},
        ]
        text = tokenizer.apply_chat_template(
            chat_template, tokenize=False, continue_final_message=True
        )
        return {"text": text}

    elif mode == "grpo":
        chat_template = [
            # {
            #     "role": "system",
            #     "content": SYSTEM_PROMPT + RESONING_ADDITION_SYSTEM_PROMT,
            # },
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        prompt_text = tokenizer.apply_chat_template(
            chat_template, tokenize=False, continue_final_message=True
        )
        return {"prompt": prompt_text, "solution": answer}
    else:
        raise ValueError(f"Unknown mode: {mode}")


def tokenize_examples(
    download_data_path: str, upload_data_path: str, num_proc: int, mode: str, model: str
):
    dataset = load_dataset(download_data_path, download_mode="force_redownload")
    tokenizer = AutoTokenizer.from_pretrained(model)

    process_example_map = partial(process_example, tokenizer=tokenizer, mode=mode)
    if isinstance(dataset, dict) and "train" in dataset:
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                process_example_map,
                num_proc=num_proc,
                desc=f"Tokenizing data for {split} split in {mode} mode",
            )
    else:
        dataset = dataset.map(
            process_example_map,
            num_proc=num_proc,
            desc=f"Tokenizing data in {mode} mode",
        )

    # Remove Unused Columns if GRPO
    column_names = set(dataset["train"].column_names)

    if mode == "grpo":
        column_names.remove("prompt")
        column_names.remove("solution")
        dataset = dataset.remove_columns(list(column_names))

    upload_data_path_with_postfix = f"{upload_data_path}_{mode}"
    dataset.push_to_hub(upload_data_path_with_postfix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT or GRPO training.")
    parser.add_argument(
        "--mode", type=str, choices=["sft", "grpo"], required=True, help="SFT or GRPO"
    )
    parser.add_argument(
        "--download_data_path",
        type=str,
        default="linuzj/graph-data-quantum",
        help="Source Dataset Path",
    )
    parser.add_argument(
        "--upload_data_path",
        type=str,
        default="Benyucong/graph-data-quantum-tokenized-4B",
        help="Tokenized Dataset Path",
    )
    parser.add_argument("--num_proc", type=int, default=20, help="Processes num.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="(default: Qwen/Qwen3-4B-Instruct-2507).",
    )
    args = parser.parse_args()

    tokenize_examples(
        download_data_path=args.download_data_path,
        upload_data_path=args.upload_data_path,
        num_proc=args.num_proc,
        mode=args.mode,
        model=args.model,
    )
