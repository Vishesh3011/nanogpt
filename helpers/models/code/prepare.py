"""
Prepare the CodeSearchNet (Go) dataset for training the code model.

Downloads `claudios/code_search_net` (Go subset), filters to functions that
have documentation and are reasonably short, formats each example as:

    <GO>
    # Function: <name>
    # Description: <doc>
    <code>
    </GO>

tokenizes with the GPT-2 BPE tokenizer, and writes `train.bin` / `val.bin`
plus `meta.pkl` (vocab size) and a `test.txt` sample for perplexity evaluation.

Usage:
    python -m models.code.prepare_data
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict

from core.tokenizer import get_tokenizer, encode

OUTPUT_DIR = os.path.join("data", "code")
MAX_CODE_CHARS = 1000


def load_codesearchnet_go() -> DatasetDict:
    """Load the Go subset of CodeSearchNet."""
    return load_dataset("claudios/code_search_net", "go")


def is_valid_example(row: dict) -> bool:
    """Keep examples that have a non-empty docstring and aren't too long."""
    doc = row.get("func_documentation_string", "")
    code = row.get("func_code_string", "")
    return len(doc.strip()) > 0 and len(code) < MAX_CODE_CHARS


def filter_dataset(dataset: DatasetDict) -> DatasetDict:
    return dataset.filter(is_valid_example)


def format_row(row: dict) -> str:
    """Format one (doc, code) pair into a <GO>...</GO> training example."""
    doc = row.get("func_documentation_string", "")
    code = row.get("func_code_string", "")
    name = row.get("func_name", "")
    return (
        "<GO>\n"
        f"# Function: {name}\n"
        f"# Description: {doc.strip()}\n"
        f"{code.strip()}\n"
        "</GO>"
    )


def get_texts(data: Dataset) -> List[str]:
    return [format_row(row) for row in data]


def save_bin(path: str, token_ids: List[int]) -> np.ndarray:
    arr = np.array(token_ids, dtype=np.uint16)
    arr.tofile(path)
    return arr


def save_meta(vocab_size: int, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "tokenizer": "gpt2"}, f)


def save_text_sample(path: str, texts: List[str], n: int = 200) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for text in texts[:n]:
            f.write(text + "\n")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_codesearchnet_go()
    dataset = filter_dataset(dataset)

    train_data = dataset["train"]
    val_data = dataset["validation"]

    print(f"train examples: {len(train_data):,}")
    print(f"val examples:   {len(val_data):,}")

    train_texts = get_texts(train_data)
    val_texts = get_texts(val_data)

    train_text = "\n".join(train_texts)
    val_text = "\n".join(val_texts)

    enc = get_tokenizer()
    train_ids = encode(train_text, enc)
    val_ids = encode(val_text, enc)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    save_bin(os.path.join(OUTPUT_DIR, "train.bin"), train_ids)
    save_bin(os.path.join(OUTPUT_DIR, "val.bin"), val_ids)
    save_meta(enc.n_vocab, os.path.join(OUTPUT_DIR, "meta.pkl"))
    save_text_sample(os.path.join(OUTPUT_DIR, "test.txt"), val_texts)

    print("Code dataset preparation complete.")


if __name__ == "__main__":
    main()
