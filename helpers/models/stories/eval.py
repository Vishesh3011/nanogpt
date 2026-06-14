"""
models/stories/eval.py

Evaluation entrypoint for the story-generation model.

Loads a trained checkpoint (or a raw GPT-2 variant) and reports average
cross-entropy loss and perplexity over a held-out paragraph file.

Usage:
    # Evaluate a trained checkpoint:
    python -m models.stories.eval --init_from=resume --out_dir=checkpoints/stories \
        --input_file=data/stories/eval.txt

    # Evaluate a GPT-2 baseline:
    python -m models.stories.eval --init_from=gpt2
"""

import os

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
init_from: str = "resume"               # "resume" | "gpt2" | "gpt2-medium" etc.
out_dir: str = "checkpoints/stories"    # where to find ckpt.pt
input_file: str = "data/stories/eval.txt"
input_format: str = "auto"             # "auto" | "txt" | "jsonl" | "json"
json_text_key: str = "text"
max_paragraphs: int = -1
print_first_n: int = 3
device: str = "cpu"
compile: bool = False
seed: int = 1337

# ---------------------------------------------------------------------------
# Apply overrides
# ---------------------------------------------------------------------------
from core.utils import apply_config_overrides  # noqa: E402
apply_config_overrides(globals())

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
import torch  # noqa: E402
from core.utils import load_checkpoint           # noqa: E402
from core.evaluator import EvalConfig, evaluate  # noqa: E402
from core.tokenizer import Tokenizer             # noqa: E402
from models.stories.model import GPT, GPTConfig  # noqa: E402

if init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gpt_config = GPTConfig(**checkpoint["config"].get("model_args", checkpoint["config"]))
    model = GPT(gpt_config)
    load_checkpoint(ckpt_path, model, device)
elif init_from.startswith("gpt2"):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"Unsupported init_from: {init_from!r}")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
tok = Tokenizer()
encode = tok.encode

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
dtype_str: str = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

eval_cfg = EvalConfig(
    input_file=input_file,
    input_format=input_format,
    json_text_key=json_text_key,
    max_paragraphs=max_paragraphs,
    print_first_n=print_first_n,
    device=device,
    dtype=dtype_str,
    compile=compile,
    seed=seed,
)

result = evaluate(model, eval_cfg, encode)
print(f"\nFinal PPL: {result.ppl:.2f}  |  avg_loss: {result.avg_loss:.4f}")