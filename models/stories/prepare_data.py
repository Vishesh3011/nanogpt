"""
Prepare the TinyStories dataset for training the stories model.

Downloads the `roneneldan/TinyStories` dataset from Hugging Face, formats
each story with GPT-2's "<|endoftext|>" token as a separator (so the model
learns to emit it at story boundaries -- used by `GPT.generate(...,
stop_at_second_eot=True)` for clean single-story sampling), tokenizes with
the GPT-2 BPE tokenizer, and writes `train.bin` / `val.bin` plus a small
`eval_stories.txt` sample for perplexity evaluation.

Usage:
    python -m models.stories.prepare_data
"""

import os
from typing import List, Tuple

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict

from core.tokenizer import get_tokenizer, encode

OUTPUT_DIR = os.path.join("data", "stories")
EOT = "<|endoftext|>"


def load_tinystories() -> Tuple[Dataset, Dataset]:
    """Load the TinyStories train/validation splits."""
    dataset: DatasetDict = load_dataset("roneneldan/TinyStories")
    return dataset["train"], dataset["validation"]


def get_stories(data: Dataset) -> List[str]:
    """Extract and strip the 'text' field from each row."""
    return [row["text"].strip() for row in data]


def format_corpus(stories: List[str]) -> str:
    """Join stories with the EOT token as a separator (and trailing EOT)."""
    return f"{EOT}\n".join(stories) + EOT


def tokenize_corpus(text: str) -> List[int]:
    enc = get_tokenizer()
    return encode(text, enc)


def save_bin(path: str, token_ids: List[int]) -> np.ndarray:
    arr = np.array(token_ids, dtype=np.uint16)
    arr.tofile(path)
    return arr


def save_eval_sample(path: str, stories: List[str], n: int = 200) -> None:
    """Save a plain-text sample of validation stories (no special tokens) for
    perplexity evaluation, paragraphs separated by blank lines."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(stories[:n]))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_data, val_data = load_tinystories()

    train_stories = get_stories(train_data)
    val_stories = get_stories(val_data)

    print(f"train stories: {len(train_stories):,}")
    print(f"val stories:   {len(val_stories):,}")

    train_text = format_corpus(train_stories)
    val_text = format_corpus(val_stories)

    train_ids = tokenize_corpus(train_text)
    val_ids = tokenize_corpus(val_text)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    save_bin(os.path.join(OUTPUT_DIR, "train.bin"), train_ids)
    save_bin(os.path.join(OUTPUT_DIR, "val.bin"), val_ids)
    save_eval_sample(os.path.join(OUTPUT_DIR, "eval_stories.txt"), val_stories)

    print("Stories dataset preparation complete.")


if __name__ == "__main__":
    main()
