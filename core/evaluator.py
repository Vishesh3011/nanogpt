"""
core/evaluator.py

Shared evaluation logic: compute average cross-entropy loss and perplexity
over a text file or JSONL file.

Both the stories and code pipelines call `evaluate()` with their respective
model and input files.  The paragraph-reading helpers support .txt, .jsonl,
and .json inputs so the same evaluator works for both datasets.

Usage:
    from core.evaluator import EvalConfig, evaluate
    result = evaluate(model, cfg)
    print(f"PPL: {result.ppl:.2f}")
"""

from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Settings for a single evaluation run."""

    input_file: str = ""
    input_format: str = "auto"  # "auto" | "txt" | "jsonl" | "json"
    json_text_key: str = "text"
    max_paragraphs: int = -1    # -1 means all
    print_first_n: int = 3      # preview first N paragraphs

    device: str = "cpu"
    dtype: str = ""             # auto-detected if empty
    compile: bool = False
    seed: int = 1337

    def __post_init__(self) -> None:
        if not self.dtype:
            self.dtype = (
                "bfloat16"
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else "float16"
            )


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Output of a single evaluate() call."""
    avg_loss: float
    ppl: float
    used_paragraphs: int
    skipped_short: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Paragraph readers
# ---------------------------------------------------------------------------

def _read_txt(path: str) -> list[str]:
    """Read paragraphs from a plain-text file (blank-line separated)."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return [p.strip() for p in content.split("\n\n") if p.strip()]


def _read_jsonl(path: str, text_key: str) -> list[str]:
    """Read paragraphs from a JSONL file (one JSON object per line)."""
    paragraphs: list[str] = []
    with open(path, encoding="utf-8") as f:
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
            if text.strip():
                paragraphs.append(text.strip())
    return paragraphs


def _read_json(path: str, text_key: str) -> list[str]:
    """Read paragraphs from a JSON file containing a list of strings or dicts."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("JSON input must be a list of strings or dicts")
    paragraphs: list[str] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            if text_key not in item:
                raise KeyError(f"Missing key '{text_key}' in JSON item {i}")
            text = item[text_key]
        else:
            raise TypeError(f"Unsupported JSON item type at index {i}: {type(item)}")
        if text.strip():
            paragraphs.append(text.strip())
    return paragraphs


def load_paragraphs(path: str, fmt: str, text_key: str) -> tuple[list[str], str]:
    """Load paragraphs from *path* in format *fmt*.

    Args:
        path:     Path to the input file.
        fmt:      "auto", "txt", "jsonl", or "json".
        text_key: Dict key to extract text from when format is jsonl/json.

    Returns:
        (paragraphs, resolved_format)
    """
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        fmt = {"txt": "txt", ".jsonl": "jsonl", ".json": "json"}.get(ext, "txt")

    if fmt == "txt":
        return _read_txt(path), "txt"
    if fmt == "jsonl":
        return _read_jsonl(path, text_key), "jsonl"
    if fmt == "json":
        return _read_json(path, text_key), "json"
    raise ValueError(f"Unsupported input_format: {fmt!r}")


# ---------------------------------------------------------------------------
# Main evaluate() function
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    cfg: EvalConfig,
    encode: Callable[[str], list[int]],
) -> EvalResult:
    """Compute average cross-entropy loss and perplexity over *cfg.input_file*.

    The function iterates over every paragraph in the file, sliding a window
    of size block_size across the tokenised paragraph and accumulating the
    per-token negative log-likelihood.  The reported perplexity is exp(mean NLL).

    Args:
        model:  A GPT model in eval mode.
        cfg:    EvalConfig specifying the input file, device, etc.
        encode: Tokeniser encode function: str -> list[int].

    Returns:
        An EvalResult with avg_loss, ppl, and bookkeeping counts.
    """
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in str(cfg.device) else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    model.eval()
    model.to(cfg.device)

    if cfg.compile:
        model = torch.compile(model)

    # Load paragraphs --------------------------------------------------
    paragraphs, used_fmt = load_paragraphs(cfg.input_file, cfg.input_format, cfg.json_text_key)
    if cfg.max_paragraphs >= 0:
        paragraphs = paragraphs[: cfg.max_paragraphs]

    if not paragraphs:
        raise ValueError(f"No paragraphs found in {cfg.input_file} (format={used_fmt})")

    print(f"Loaded {len(paragraphs)} paragraphs from {cfg.input_file} (format={used_fmt})")
    for i, p in enumerate(paragraphs[: max(0, cfg.print_first_n)]):
        preview = p.replace("\n", " ")[:120]
        print(f"[preview {i}] {preview}{'...' if len(p) > 120 else ''}")

    # Evaluate ---------------------------------------------------------
    total_nll = 0.0
    total_tokens = 0
    used_paragraphs = 0
    skipped_short = 0
    block_size: int = model.config.block_size  # type: ignore[attr-defined]

    with torch.no_grad():
        with ctx:
            for para in paragraphs:
                token_ids = encode(para)
                # Need at least 2 tokens for next-token prediction.
                if len(token_ids) < 2:
                    skipped_short += 1
                    continue

                pos = 0
                n_pred = len(token_ids) - 1
                while pos < n_pred:
                    inp = token_ids[pos : pos + block_size]
                    tgt = token_ids[pos + 1 : pos + 1 + block_size]
                    if not tgt:
                        break
                    if len(inp) != len(tgt):
                        inp = inp[: len(tgt)]

                    x = torch.tensor(inp, dtype=torch.long, device=cfg.device)[None, :]
                    y = torch.tensor(tgt, dtype=torch.long, device=cfg.device)[None, :]
                    _, loss = model(x, y)

                    n_tok = len(tgt)
                    total_nll += loss.item() * n_tok
                    total_tokens += n_tok
                    pos += n_tok

                used_paragraphs += 1

    if total_tokens == 0:
        raise ValueError("No valid tokens to evaluate. Check your input file.")

    avg_loss = total_nll / total_tokens
    ppl = math.exp(avg_loss)

    print("----- Evaluation Results -----")
    print(f"paragraphs_used : {used_paragraphs}")
    print(f"paragraphs_skip : {skipped_short}")
    print(f"pred_tokens     : {total_tokens}")
    print(f"avg_loss        : {avg_loss:.4f}")
    print(f"ppl             : {ppl:.2f}")

    return EvalResult(
        avg_loss=avg_loss,
        ppl=ppl,
        used_paragraphs=used_paragraphs,
        skipped_short=skipped_short,
        total_tokens=total_tokens,
    )