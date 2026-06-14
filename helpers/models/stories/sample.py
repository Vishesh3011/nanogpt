"""
Generate sample stories from a trained stories model.

Usage:
    python -m models.stories.sample
    python -m models.stories.sample --num_samples=5 --max_new_tokens=300
    python -m models.stories.sample --prompt="Once upon a time"
"""

import json
import os

import torch

from core.evaluator import generate_text, load_model_from_checkpoint, load_pretrained_gpt2
from core.tokenizer import get_eot_token_id, get_tokenizer
from core.utils import default_dtype, load_config, seed_everything, setup_device

DEFAULT_CONFIG = {
    "init_from": "resume",  # 'resume' or a GPT-2 variant e.g. 'gpt2'
    "out_dir": os.path.join("checkpoints", "stories"),
    "device": "cpu",
    "dtype": default_dtype(),
    "compile": False,
    "seed": 1337,

    # generation
    "prompt": "\n",  # start of sequence; stories begin right after an EOT token
    "num_samples": 5,
    "max_new_tokens": 300,
    "temperature": 0.8,
    "top_k": 200,

    # output
    "output_file": "samples.jsonl",  # set to None / '' to disable
}


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)

    seed_everything(cfg["seed"])
    _, _, ctx = setup_device(cfg["device"], cfg["dtype"])

    if cfg["init_from"] == "resume":
        model = load_model_from_checkpoint(cfg["out_dir"], device=cfg["device"])
    elif cfg["init_from"].startswith("gpt2"):
        model = load_pretrained_gpt2(cfg["init_from"])
        model.to(cfg["device"])
    else:
        raise ValueError(f"Unsupported init_from: {cfg['init_from']}")

    if cfg["compile"]:
        model = torch.compile(model)

    enc = get_tokenizer()
    eot_id = get_eot_token_id(enc)

    output_f = open(cfg["output_file"], "w", encoding="utf-8") if cfg["output_file"] else None

    with torch.no_grad():
        with ctx:
            for i in range(cfg["num_samples"]):
                story = generate_text(
                    model, cfg["prompt"], enc, cfg["device"],
                    max_new_tokens=cfg["max_new_tokens"],
                    temperature=cfg["temperature"],
                    top_k=cfg["top_k"],
                    eot_token=eot_id,
                    stop_at_second_eot=True,
                )
                print(story)
                print("---------------")

                if output_f:
                    output_f.write(json.dumps({
                        "prompt": cfg["prompt"],
                        "generated_text": story,
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