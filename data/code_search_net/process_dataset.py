from typing import Dict, List

from datasets import load_dataset
from datasets import Dataset, DatasetDict

def load_data_from_repo() -> DatasetDict:
    dataset: DatasetDict = load_dataset("claudios/code_search_net", "go")
    return dataset

def is_valid_example(example: Dict) -> bool:
    doc: str = example.get("func_documentation_string", "")
    code: str = example.get("func_code_string", "")

    return len(doc.strip()) > 0 and len(code) < 1000


def filter_dataset(dataset: DatasetDict) -> DatasetDict:
    return dataset.filter(is_valid_example)

def format_example(example: Dict) -> Dict[str, str]:
    doc: str = example.get("func_documentation_string", "")
    code: str = example.get("func_code_string", "")
    name: str = example.get("func_name", "")

    text: str = (
        "<GO>\n"
        f"# Function: {name}\n"
        f"# Description: {doc.strip()}\n"
        f"{code.strip()}\n"
        "</GO>"
    )

    return {"text": text}

def format_dataset(dataset: DatasetDict) -> DatasetDict:
    return dataset.map(format_example)

def print_debug_info(dataset: DatasetDict) -> None:
    print("Train samples:", len(dataset["train"]))
    print("Validation samples:", len(dataset["validation"]))
    print("Test samples:", len(dataset["test"]))

    sample: str = dataset["train"][0]["text"]
    print("\nSample:\n", sample[:300])

def keep_only_text(dataset: DatasetDict) -> DatasetDict:
    for split in dataset.keys():
        dataset[split] = dataset[split].remove_columns(
            [col for col in dataset[split].column_names if col != "text"]
        )
    return dataset

def save_json(dataset: DatasetDict, base_path: str) -> None:
    dataset["train"].to_json(f"{base_path}/train.json")
    dataset["validation"].to_json(f"{base_path}/validation.json")
    dataset["test"].to_json(f"{base_path}/test.json")


def save_txt(dataset: DatasetDict, base_path: str) -> None:
    with open(f"{base_path}/train.txt", "w") as f:
        for ex in dataset["train"]:
            f.write(ex["text"] + "\n")

    with open(f"{base_path}/val.txt", "w") as f:
        for ex in dataset["validation"]:
            f.write(ex["text"] + "\n")

if __name__ == "__main__":
    dataset: DatasetDict = load_data_from_repo()

    dataset = filter_dataset(dataset)
    dataset = format_dataset(dataset)

    print_debug_info(dataset)

    dataset = keep_only_text(dataset)

    base_path: str = "data/code_search_net"

    save_json(dataset, base_path)
    save_txt(dataset, base_path)

    print("Preprocessing complete.")