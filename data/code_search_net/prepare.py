import os
import pickle
from typing import List, Tuple

import numpy as np
import tiktoken

from datasets import load_dataset
from datasets import Dataset, DatasetDict

def load_data_from_repo() -> DatasetDict:
    dataset: DatasetDict = load_dataset("claudios/code_search_net", "go")
    return dataset

def is_valid_example(row: dict) -> bool:
    doc: str = row.get("func_documentation_string", "")
    code: str = row.get("func_code_string", "")

    return len(doc.strip()) > 0 and len(code) < 1000


def filter_dataset(dataset: DatasetDict) -> DatasetDict:
    return dataset.filter(is_valid_example)


def format_row(row: dict) -> str:
    doc: str = row.get("func_documentation_string", "")
    code: str = row.get("func_code_string", "")
    name: str = row.get("func_name", "")

    return (
        "<GO>\n"
        f"# Function: {name}\n"
        f"# Description: {doc.strip()}\n"
        f"{code.strip()}\n"
        "</GO>"
    )


def get_texts(data: Dataset) -> List[str]:
    texts: List[str] = []

    for row in data:
        formatted: str = format_row(row)
        texts.append(formatted)

    return texts

def get_text(texts: List[str]) -> str:
    return "\n".join(texts)

def get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")

def get_encodings(text: str, enc: tiktoken.Encoding) -> List[int]:
    return enc.encode(text)

def split_tokens(tokens: List[int], split_ratio: float = 0.9) -> Tuple[List[int], List[int]]:
    split_idx: int = int(len(tokens) * split_ratio)

    train_ids: List[int] = tokens[:split_idx]
    val_ids: List[int] = tokens[split_idx:]

    return train_ids, val_ids

def save_bin_files(train_ids: List[int], val_ids: List[int], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    train_array: np.ndarray = np.array(train_ids, dtype=np.uint16)
    val_array: np.ndarray = np.array(val_ids, dtype=np.uint16)

    train_array.tofile(os.path.join(output_dir, "train.bin"))
    val_array.tofile(os.path.join(output_dir, "val.bin"))

def save_meta(enc: tiktoken.Encoding, output_dir: str) -> None:
    meta: dict = {
        "vocab_size": enc.n_vocab,
        "tokenizer": "gpt2"
    }

    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    dataset: DatasetDict = load_data_from_repo()

    dataset = filter_dataset(dataset)

    train_data: Dataset = dataset["train"]
    val_data: Dataset = dataset["validation"]

    train_texts: List[str] = get_texts(train_data)
    val_texts: List[str] = get_texts(val_data)

    train_text: str = get_text(train_texts)
    val_text: str = get_text(val_texts)

    encoder: tiktoken.Encoding = get_encoder()

    train_tokens: List[int] = get_encodings(train_text, encoder)
    val_tokens: List[int] = get_encodings(val_text, encoder)

    print(f"train has {len(train_tokens):,} tokens")
    print(f"val has {len(val_tokens):,} tokens")

    output_dir: str = "data/code_search_net"

    save_bin_files(train_tokens, val_tokens, output_dir)
    save_meta(encoder, output_dir)

    print("Dataset preparation complete.")