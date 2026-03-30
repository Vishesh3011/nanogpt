"""
Evaluate GPT/GPT-2 on a paragraph file and report average loss and perplexity.

Supported input formats:
- .txt: paragraphs separated by one or more blank lines
- .jsonl: one JSON object per line, text field configurable by `json_text_key`
- .json: list[str] or list[dict], text field configurable by `json_text_key`
"""

import json
import math
import os
import pickle
from contextlib import nullcontext

import torch
import tiktoken

from model2 import GPT, GPTConfig

# -----------------------------------------------------------------------------
# model/load config (same pattern as sample.py)
init_from = 'gpt2'  # 'resume' or a GPT-2 variant (e.g. 'gpt2-medium')
out_dir = 'out'  # used when init_from == 'resume'
device = torch.device("cpu")  # 'cpu', 'cuda', 'cuda:0', ...
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
seed = 1337

# data/eval config
input_file = 'data/rocstories/eval_stories.txt'
input_format = 'txt'  # 'auto' | 'txt' | 'jsonl' | 'json'
json_text_key = 'text'
max_paragraphs = -1  # -1 means all
print_first_n = 3  # preview first N loaded paragraphs

exec(open('configurator.py').read())  # allows overrides from CLI / config file
# -----------------------------------------------------------------------------


def _read_txt_paragraphs(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    parts = content.split('\n\n')
    return [p.strip() for p in parts if p.strip()]


def _read_jsonl_paragraphs(path, text_key):
    paragraphs = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                text = obj
            elif isinstance(obj, dict):
                if text_key not in obj:
                    raise KeyError(f"Missing key '{text_key}' in JSONL line {ln}")
                text = obj[text_key]
            else:
                raise TypeError(f"Unsupported JSONL value type on line {ln}: {type(obj)}")
            text = text.strip()
            if text:
                paragraphs.append(text)
    return paragraphs


def _read_json_paragraphs(path, text_key):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("JSON input must be a list of strings or objects")
    paragraphs = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            if text_key not in item:
                raise KeyError(f"Missing key '{text_key}' in JSON item index {i}")
            text = item[text_key]
        else:
            raise TypeError(f"Unsupported JSON item type at index {i}: {type(item)}")
        text = text.strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def load_paragraphs(path, fmt, text_key):
    if fmt == 'auto':
        ext = os.path.splitext(path)[1].lower()
        if ext == '.txt':
            fmt = 'txt'
        elif ext == '.jsonl':
            fmt = 'jsonl'
        elif ext == '.json':
            fmt = 'json'
        else:
            fmt = 'txt'

    if fmt == 'txt':
        return _read_txt_paragraphs(path), 'txt'
    if fmt == 'jsonl':
        return _read_jsonl_paragraphs(path, text_key), 'jsonl'
    if fmt == 'json':
        return _read_json_paragraphs(path, text_key), 'json'
    raise ValueError(f"Unsupported input_format: {fmt}")


torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# device_type = 'cuda' if 'cuda' in device else 'cpu'
device_type = device.type
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"Unsupported init_from: {init_from}")

model.eval()

model.to(device)
if compile:
    model = torch.compile(model)

# tokenizer (same behavior as sample.py)
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    encode = lambda s: [stoi[c] for c in s]
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

paragraphs, used_fmt = load_paragraphs(input_file, input_format, json_text_key)
if max_paragraphs is not None and max_paragraphs >= 0:
    paragraphs = paragraphs[:max_paragraphs]

if len(paragraphs) == 0:
    raise ValueError(f"No paragraphs found in {input_file} (format={used_fmt})")

print(f"Loaded {len(paragraphs)} paragraphs from {input_file} (format={used_fmt})")
for i, p in enumerate(paragraphs[:max(0, int(print_first_n))]):
    preview = p.replace('\n', ' ')[:120]
    print(f"[preview {i}] {preview}{'...' if len(p) > 120 else ''}")

total_nll = 0.0
total_tokens = 0
used_paragraphs = 0
skipped_short = 0
block_size = model.config.block_size

with torch.no_grad():
    with ctx:
        for para in paragraphs:
            token_ids = encode(para)
            # Need at least two tokens to define next-token prediction.
            if len(token_ids) < 2:
                skipped_short += 1
                continue

            pos = 0
            para_pred_tokens = len(token_ids) - 1
            while pos < para_pred_tokens:
                # Build a contiguous chunk and its shifted targets.
                inp = token_ids[pos: pos + block_size]
                tgt = token_ids[pos + 1: pos + 1 + block_size]
                if len(tgt) == 0:
                    break
                if len(inp) != len(tgt):
                    inp = inp[:len(tgt)]

                x = torch.tensor(inp, dtype=torch.long, device=device)[None, :]
                y = torch.tensor(tgt, dtype=torch.long, device=device)[None, :]
                _, loss = model(x, y)  # mean CE over chunk tokens

                n_tok = len(tgt)
                total_nll += loss.item() * n_tok
                total_tokens += n_tok
                pos += n_tok

            used_paragraphs += 1

if total_tokens == 0:
    raise ValueError("No valid tokens to evaluate. Check your input text.")

avg_loss = total_nll / total_tokens
ppl = math.exp(avg_loss)

print("----- Evaluation Results -----")
print(f"model           : {init_from}")
print(f"paragraphs_used : {used_paragraphs}")
print(f"paragraphs_skip : {skipped_short}")
print(f"pred_tokens     : {total_tokens}")
print(f"avg_loss        : {avg_loss:.3f}")
print(f"ppl             : {ppl:.2f}")

if __name__ == '__main__':
    print("Evaluation complete.")
    pass