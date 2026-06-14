import tiktoken
import numpy as np
from datasets import load_dataset, Dataset


# =========================
# DATA LOADING
# =========================
def load_data_from_repo() -> tuple[Dataset, Dataset]:
    dataset = load_dataset("mintujupally/ROCStories")

    # IMPORTANT: shuffle must be reassigned
    dataset = dataset.shuffle(seed=42)

    train_data = dataset["train"]
    test_data = dataset["test"]

    return train_data, test_data


# =========================
# STORY EXTRACTION
# =========================
def get_story_from_row(row) -> str:
    return row["text"].strip()


def get_stories(data) -> list[str]:
    return [get_story_from_row(row) for row in data]


# =========================
# TEXT FORMATTING
# =========================

# --- Option A (minimal, safe baseline) ---
def format_train_text_basic(stories: list[str]) -> str:
    return "<|endoftext|>\n".join(stories) + "<|endoftext|>"


def format_test_text_basic(stories: list[str]) -> str:
    # NO special tokens in test
    return "\n\n".join(stories)


# --- Option B (recommended: structured prompts for better generation) ---
def format_train_text_structured(stories: list[str]) -> str:
    formatted = []
    for story in stories:
        formatted.append(f"<|endoftext|>\nStory:\n{story}\n")
    return "".join(formatted) + "<|endoftext|>"


def format_test_text_structured(stories: list[str]) -> str:
    # Keep test natural — no special tokens
    return "\n\n".join(stories)


# =========================
# TOKENIZATION
# =========================
def get_encodings(text: str) -> list[int]:
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(text, allowed_special={"<|endoftext|>"})


# =========================
# SAVE FILES
# =========================
def save_text_file(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_bin_file(path: str, token_ids: list[int]):
    arr = np.array(token_ids, dtype=np.uint16)
    arr.tofile(path)
    return arr


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":
    train_data, test_data = load_data_from_repo()

    train_stories = get_stories(train_data)
    test_stories = get_stories(test_data)

    # -------- CHOOSE FORMAT --------
    USE_STRUCTURED = True

    if USE_STRUCTURED:
        train_text = format_train_text_structured(train_stories)
        test_text = format_test_text_structured(test_stories)
    else:
        train_text = format_train_text_basic(train_stories)
        test_text = format_test_text_basic(test_stories)

    # -------- SAVE RAW TEXT --------
    save_text_file("data/rocstories/train.txt", train_text)
    save_text_file("data/rocstories/test.txt", test_text)

    # -------- TOKENIZE --------
    train_encodings = get_encodings(train_text)
    test_encodings = get_encodings(test_text)

    # -------- SAVE BIN FILES --------
    train_ids = save_bin_file("data/rocstories/train.bin", train_encodings)
    val_ids = save_bin_file("data/rocstories/val.bin", test_encodings)

    # -------- STATS --------
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens:   {len(val_ids):,}")