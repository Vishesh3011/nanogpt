from typing import List, Tuple

import math
import os
import pickle

import torch
import tiktoken

from model2 import GPT, GPTConfig

def load_model(out_dir: str, device: torch.device) -> GPT:
    ckpt_path: str = os.path.join(out_dir, "ckpt.pt")
    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)

    state_dict = checkpoint["model"]

    unwanted_prefix: str = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model

def get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")

def encode_text(text: str, enc: tiktoken.Encoding) -> List[int]:
    return enc.encode(text)

def decode_tokens(tokens: List[int], enc: tiktoken.Encoding) -> str:
    return enc.decode(tokens)

def compute_loss(
    model: GPT,
    tokens: List[int],
    block_size: int,
    device: torch.device
) -> Tuple[float, int]:
    total_loss: float = 0.0
    total_tokens: int = 0

    pos: int = 0

    while pos < len(tokens) - 1:
        inp = tokens[pos: pos + block_size]
        tgt = tokens[pos + 1: pos + 1 + block_size]

        if len(tgt) == 0:
            break

        if len(inp) != len(tgt):
            inp = inp[:len(tgt)]

        x = torch.tensor(inp, dtype=torch.long, device=device)[None, :]
        y = torch.tensor(tgt, dtype=torch.long, device=device)[None, :]

        with torch.no_grad():
            _, loss = model(x, y)

        n: int = len(tgt)
        total_loss += loss.item() * n
        total_tokens += n

        pos += n

    return total_loss, total_tokens

def load_test_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def compute_perplexity(
    model: GPT,
    text: str,
    enc: tiktoken.Encoding,
    block_size: int,
    device: torch.device
) -> Tuple[float, float]:
    tokens: List[int] = encode_text(text, enc)

    total_loss, total_tokens = compute_loss(model, tokens, block_size, device)

    avg_loss: float = total_loss / total_tokens
    ppl: float = math.exp(avg_loss)

    return avg_loss, ppl

def generate_text(
    model: GPT,
    prompt: str,
    enc: tiktoken.Encoding,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200
) -> str:
    tokens: List[int] = encode_text(prompt, enc)

    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        # out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
        out = model.generate(idx, max_new_tokens = max_new_tokens, temperature = temperature, top_k = top_k)

    return decode_tokens(out[0].tolist(), enc)

if __name__ == "__main__":
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir: str = "out-rocstories"

    model: GPT = load_model(out_dir, device)
    enc: tiktoken.Encoding = get_encoder()

    block_size: int = model.config.block_size

    test_txt = load_test_text("data/code_search_net/test.txt")
    avg_loss, ppl = compute_perplexity(model, test_txt, enc, block_size, device)
    
    print
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")

    prompt: str = "<GO>\n# Function: reverse\n# Description: reverse a string\nfunc reverse(s string) \n"

    generated: str = generate_text(model, prompt, enc, device,max_new_tokens = 80, temperature = 0.6, top_k = 40)

    # truncate at first </GO>
    if "</GO>" in generated:
        generated = generated.split("</GO>")[0] + "</GO>"

    print("\n----- Generated Output -----")
    print(generated)