"""
Generate Go code samples from a trained code model, optionally batching
multiple prompts from a file.

Usage:
    python -m models.code.sample
    python -m models.code.sample --prompt="<GO>\\n# Function: max\\n# Description: return the max of two ints\\n"
    python -m models.code.sample --start="FILE:data/code/prompts.txt" --batch_prompts=True
"""

import json
import os

import torch

from core.evaluator import generate_text, load_model_from_checkpoint
from core.tokenizer import get_tokenizer
from core.utils import default_dtype, load_config, seed_everything, setup_device

DEFAULT_CONFIG = {
    "out_dir": os.path.join("checkpoints", "code"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": default_dtype(),
    "compile": False,
    "seed": 1337,

    # prompt(s): either a literal string, or "FILE:path/to/file.txt"
    "start": "<GO>\n# Function: add\n# Description: add two numbers\nfunc add(a int, b int) \n",
    "batch_prompts": False,  # if True and start is a FILE:, read one prompt per line

    # generation
    "num_samples": 1,
    "max_new_tokens": 128,
    "temperature": 0.6,
    "top_k": 40,

    # output
    "output_file": "samples.jsonl",  # set to None / '' to disable
}


def _load_prompts(start: str, batch_prompts: bool) -> list:
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            if batch_prompts:
                return [line.rstrip() for line in f.readlines() if line.strip()]
            return [f.read()]
    return [start]


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)

    seed_everything(cfg["seed"])
    _, _, ctx = setup_device(cfg["device"], cfg["dtype"])

    model = load_model_from_checkpoint(cfg["out_dir"], device=cfg["device"])
    if cfg["compile"]:
        model = torch.compile(model)

    enc = get_tokenizer()
    prompts = _load_prompts(cfg["start"], cfg["batch_prompts"])

    output_f = open(cfg["output_file"], "w", encoding="utf-8") if cfg["output_file"] else None

    with torch.no_grad():
        with ctx:
            for prompt_idx, prompt in enumerate(prompts):
                if cfg["batch_prompts"] and len(prompts) > 1:
                    print(f"\n=== Prompt {prompt_idx + 1}: {prompt} ===")

                for _ in range(cfg["num_samples"]):
                    generated = generate_text(
                        model, prompt, enc, cfg["device"],
                        max_new_tokens=cfg["max_new_tokens"],
                        temperature=cfg["temperature"],
                        top_k=cfg["top_k"],
                    )
                    if "</GO>" in generated:
                        generated = generated.split("</GO>")[0] + "</GO>"

                    print(generated)
                    print("---------------")

                    if output_f:
                        output_f.write(json.dumps({
                            "prompt": prompt,
                            "generated_text": generated,
                            "params": {
                                "max_new_tokens": cfg["max_new_tokens"],
                                "temperature": cfg["temperature"],
                                "top_k": cfg["top_k"],
                            },
                        }) + "\n")

    if output_f:
        output_f.close()
        print(f"\nResults saved to {cfg['output_file']}")


if __name__ == "__main__":
    main()
