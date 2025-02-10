from typing import Dict
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_hypergraph_example(example: Dict) -> Dict:
    n_qubits = example["number_of_qubits"]
    n_layers = example["number_of_layers"]
    hypergraph = example["hypergraph"]
    circuit_with_params = example["circuit_with_params"]
    circuit_with_symbols = example["circuit_with_symbols"]

    question = f"Generate a quantum circuit with {n_qubits} qubits and {n_layers} layers to solve the hypergraph max-cut problem using VQE with the following hypergraph: {hypergraph}."
    answer = circuit_with_params

    return dict(
        question=question,
        answer=answer,
        circuit_with_params=circuit_with_params,
        circuit_with_symbols=circuit_with_symbols,
    )


def process_example(
    example: Dict,
    tokenizer,
):
    question, answer, circuit_with_params, circuit_with_symbols = (
        process_hypergraph_example(example).values()
    )
    prompt = QUERY_TEMPLATE_NOANSWER.format(Question=question)
    answer = "Answer: " + answer if "Answer:" not in answer else answer
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "<|im_start|>think\n"
                + "\n<|im_start|>answer\n"
                + answer.strip(),
            },
        ],
        tokenize=False,
    )
    return dict(text=text)


def tokenize_examples_for_sft(
    upload_data_path: str,
    download_data_path: str,
    num_proc: int,
):
    dataset = load_dataset(download_data_path, download_mode="force_redownload")

    if "train" in dataset:
        dataset = dataset["train"]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    process_example_map = partial(process_example, tokenizer=tokenizer)
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
