from datasets import load_dataset

# Load dataset
dataset = load_dataset("claudios/code_search_net", "go")

# -----------------------------
# Filtering (less aggressive)
# -----------------------------
def filter_example(example):
    doc = example.get("func_documentation_string", "")
    code = example.get("func_code_string", "")
    
    return len(doc.strip()) > 0 and len(code) < 1000  # increased limit


# -----------------------------
# Formatting (text -> code)
# -----------------------------
def format_example(example):
    doc = example.get("func_documentation_string", "")
    code = example.get("func_code_string", "")
    name = example.get("func_name", "")

    return {
        "text": f"<GO>\n# Function: {name}\n# Description: {doc.strip()}\n{code.strip()}\n</GO>"
    }


# Apply transformations
dataset = dataset.filter(filter_example)
dataset = dataset.map(format_example)

# -----------------------------
# Debug (IMPORTANT)
# -----------------------------
print("Train samples:", len(dataset["train"]))
print("Validation samples:", len(dataset["validation"]))
print("Test samples:", len(dataset["test"]))

print("\nSample:\n", dataset["train"][0]["text"][:300])


# -----------------------------
# Keep only "text" column
# -----------------------------
for split in dataset.keys():
    dataset[split] = dataset[split].remove_columns(
        [col for col in dataset[split].column_names if col != "text"]
    )


# -----------------------------
# Save JSON (your current setup)
# -----------------------------
dataset["train"].to_json("data/code_search_net/train.json")
dataset["validation"].to_json("data/code_search_net/validation.json")
dataset["test"].to_json("data/code_search_net/test.json")


# -----------------------------
# ALSO save TXT (recommended for NanoGPT)
# -----------------------------
with open("data/code_search_net/train.txt", "w") as f:
    for ex in dataset["train"]:
        f.write(ex["text"] + "\n")

with open("data/code_search_net/val.txt", "w") as f:
    for ex in dataset["validation"]:
        f.write(ex["text"] + "\n")

print("\n✅ Preprocessing complete.")