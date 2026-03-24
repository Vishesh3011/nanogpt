import os
import pickle
import numpy as np
import tiktoken

# -----------------------------
# Input
# -----------------------------
input_file = "data/code_search_net/train.txt"

with open(input_file, "r", encoding="utf-8") as f:
    data = f.read()

# -----------------------------
# Tokenizer (GPT-2 BPE)
# -----------------------------
enc = tiktoken.get_encoding("gpt2")

tokens = enc.encode(data)

print(f"Total tokens: {len(tokens):,}")

# -----------------------------
# Train/Val split
# -----------------------------
split = int(0.9 * len(tokens))
train_ids = tokens[:split]
val_ids = tokens[split:]

# -----------------------------
# Save as uint16
# -----------------------------
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

os.makedirs("data/code_search_net", exist_ok=True)

train_ids.tofile("data/code_search_net/train.bin")
val_ids.tofile("data/code_search_net/val.bin")

# -----------------------------
# Meta info
# -----------------------------
meta = {
    "vocab_size": enc.n_vocab,
    "tokenizer": "gpt2"
}

with open("data/code_search_net/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("✅ LLaMA-style dataset prepared (GPT2 tokenizer)")