from enum import Enum
from typing import Dict
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()


class CommunityDetectionAttributes:
    communities_size: int
    number_of_communities: int


class ConnectedComponentAttributes:
    node: str


class GraphColoringAttributes:
    number_of_colors: int


class GraphIsomorphismAttributes:
    number_of_colors: int


class OptimizationProblemType(str, Enum):
    """
    Enum class representing different types of optimization problems.

    Attributes:
        HYPERGRAPH_CUT (str): Represents the hypergraph cut optimization problem type.
    """

    CONNECTED_COMPONENTS = "connected_components"
    COMMUNITY_DETECTION = "community_detection"
    K_CLIQUE = "kclique"
    HYPERMAXCUT = "hypermaxcut"
    GRAPH_ISOMORPHISM = "graph_isomorphism"
    GRAPH_COLORING = "graph_coloring"


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text


def generate_problem_specific_text(
    problem: OptimizationProblemType, attributes: Dict
) -> str:
    if problem == OptimizationProblemType.COMMUNITY_DETECTION:
        return f"with {attributes['community_size']} sized communities and {attributes['number_of_communities']} communities"
    elif problem == OptimizationProblemType.CONNECTED_COMPONENTS:
        return f"with {attributes['nodes']} nodes"
    elif problem == OptimizationProblemType.GRAPH_COLORING:
        return f"with {attributes['number_of_colors']} colors"
    elif problem == OptimizationProblemType.GRAPH_ISOMORPHISM:
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

    problem_specific_text = generate_problem_specific_text(
        problem_type, example["problem_specific_attributes"]
    )
    question = (
        f"Your task is to generate a quantum circuit in QASM 3.0 with {n_qubits} qubits and {n_layers} "
        f" layers with optimal parameters that solves the {problem_type} {problem_specific_text} for "
        f"the following graph: {graph}. Then ensure that the final answer is correct and in valid QASM 3.0 code."
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


def process_example(example: Dict, tokenizer):
    graph_data = process_graph_example(example)
    question = graph_data["question"]
    answer = graph_data["answer"]

    if "Answer:" not in answer:
        answer = "Answer: " + answer

    prompt = QUERY_TEMPLATE_NOANSWER.format(Question=question)

    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "\n<|im_start|>\n" + answer.strip(),
            },
        ],
        tokenize=False,
    )
    return dict(text=text)


def tokenize_examples_for_sft(
    upload_data_path: str, download_data_path: str, num_proc: int
):
    dataset = load_dataset(download_data_path, download_mode="force_redownload")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    process_example_map = staticmethod(partial(process_example, tokenizer=tokenizer))

    # If the dataset is a DatasetDict with splits (e.g., "train" and "test"),
    # process each split separately.
    if isinstance(dataset, dict) and "train" in dataset:
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                process_example_map,
                num_proc=num_proc,
                desc=f"Tokenizing SFT data for {split} split",
            )
    else:
        dataset = dataset.map(
            process_example_map,
            num_proc=num_proc,
            desc="Tokenizing SFT data",
        )

    dataset.push_to_hub(upload_data_path)


if __name__ == "__main__":
    tokenize_examples_for_sft(
        download_data_path="linuzj/hypergraph-max-cut-quantum",
        upload_data_path="linuzj/hypergraph-max-cut-quantum_tokenized",
        num_proc=20,
    )
