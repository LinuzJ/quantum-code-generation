import pandas as pd
from transformers import pipeline, Pipeline

CIRCUITS_FOLDER_PATH = "src/data/benchmark_mapping.csv"

pipe = pipeline(
    "text-generation", model="deepseek-ai/DeepSeek-R1", trust_remote_code=True, device=0
)


def get_mapping_table(path: str) -> pd.DataFrame:
    """Read the benchmark mapping table."""
    return pd.read_csv(path, sep="§")


df = get_mapping_table(CIRCUITS_FOLDER_PATH)


def generate_synthetic_code(row: pd.Series, max_length: int, pipeline: Pipeline) -> str:
    """Generate synthetic code from QASM."""

    messages = [
        {
            "role": "user",
            "content": f"Generate CirQ code that implements the {row['Algorithm']} algorithm on {row['Qubits']} qubits. Here is a more detailed description of the algorithm: {row['Description']}. It should replicate this QASM code: {row['QASM']}",
        },
    ]

    return pipeline(messages, max_length=max_length)


df["cirq"] = df.apply(lambda row: generate_synthetic_code(row, 100, pipe), axis=1)
