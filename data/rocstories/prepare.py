import tiktoken
import numpy as np

from datasets import load_dataset
from datasets import Dataset

def load_data_from_repo() -> tuple[Dataset, Dataset]:
    dataset = load_dataset("mintujupally/ROCStories")
    train_data = []
    test_data = []
    train_data = dataset['train']
    test_data = dataset['test']
    return (train_data, test_data)

def get_story_from_row(row) -> str:
    return row["text"]

def get_stories(data) -> tuple[list[str]]:
    stories = []
    for row in data:
        story = get_story_from_row(row)
        stories.append(story)
    return stories

def get_text(stories) -> str:
    return "\n\n".join(stories)

def get_encodings(text: str) -> list[int]:
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(text)

if __name__ == "__main__":
    train_data, test_data = load_data_from_repo()

    train_stories = get_stories(train_data)
    test_stories = get_stories(test_data)

    train_text = get_text(train_stories)
    test_text = get_text(test_stories)

    train_encodings = get_encodings(train_text)
    test_encodings = get_encodings(test_text)

    train_ids = np.array(train_encodings, dtype=np.uint16)
    val_ids = np.array(test_encodings, dtype=np.uint16)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids.tofile('data/rocstories/train.bin')
    val_ids.tofile('data/rocstories/val.bin')
