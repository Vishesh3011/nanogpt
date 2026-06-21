"""
Evaluate the code model: report perplexity on held-out Go functions and print
a sample code generation for a given prompt.

Usage:
    python -m models.code.eval
    python -m models.code.eval --out_dir=checkpoints/code --input_file=data/code/test.txt
"""

import os

import torch

from core.evaluator import compute_perplexity, generate_text, load_model_from_checkpoint
from core.tokenizer import get_tokenizer
from core.utils import default_dtype, load_config, setup_device

DEFAULT_CONFIG = {
    "out_dir": os.path.join("checkpoints", "code"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": default_dtype(),
    "compile": False,
    "seed": 1337,

    # eval data
    "input_file": os.path.join("data", "code", "test.txt"),

    # sample generation
    "prompt": "<GO>\n# Function: add\n# Description: add two numbers\nfunc add(a int, b int) \n",
    "max_new_tokens": 80,
    "temperature": 0.6,
    "top_k": 40,
}


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)

    torch.manual_seed(cfg["seed"])
    _, _, ctx = setup_device(cfg["device"], cfg["dtype"])

    model = load_model_from_checkpoint(cfg["out_dir"], device=cfg["device"])
    if cfg["compile"]:
        model = torch.compile(model)

    enc = get_tokenizer()

    with open(cfg["input_file"], "r", encoding="utf-8") as f:
        test_text = f.read()

    with ctx:
        avg_loss, ppl = compute_perplexity(model, test_text, enc, model.config.block_size, cfg["device"])

    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")

    with ctx:
        generated = generate_text(
            model, cfg["prompt"], enc, cfg["device"],
            max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"], top_k=cfg["top_k"],
        )

    # truncate at the first closing </GO> tag for a clean sample
    if "</GO>" in generated:
        generated = generated.split("</GO>")[0] + "</GO>"

    print("\n----- Generated Output -----")
    print(generated)


if __name__ == "__main__":
    main()
