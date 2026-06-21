"""
Evaluate the stories model: report perplexity on a held-out set of stories
and print a sample generation.

Usage:
    python -m models.stories.eval
    python -m models.stories.eval --out_dir=checkpoints/stories --input_file=data/stories/eval_stories.txt
"""

import os

import torch

from core.evaluator import (
    compute_perplexity_over_paragraphs,
    generate_text,
    load_model_from_checkpoint,
    load_pretrained_gpt2,
    load_paragraphs,
)
from core.tokenizer import get_eot_token_id, get_tokenizer
from core.utils import default_dtype, load_config, setup_device

DEFAULT_CONFIG = {
    "init_from": "resume",  # 'resume' or a GPT-2 variant e.g. 'gpt2'
    "out_dir": os.path.join("checkpoints", "stories"),
    "device": "cpu",
    "dtype": default_dtype(),
    "compile": False,
    "seed": 1337,

    # eval data
    "input_file": os.path.join("data", "stories", "eval_stories.txt"),
    "input_format": "auto",
    "json_text_key": "text",
    "max_paragraphs": -1,
    "print_first_n": 3,

    # sample generation
    "prompt": "\n",
    "max_new_tokens": 200,
    "temperature": 0.8,
    "top_k": 200,
}


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)

    torch.manual_seed(cfg["seed"])
    device_type, _, ctx = setup_device(cfg["device"], cfg["dtype"])

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

    paragraphs, used_fmt = load_paragraphs(cfg["input_file"], cfg["input_format"], cfg["json_text_key"])
    if cfg["max_paragraphs"] is not None and cfg["max_paragraphs"] >= 0:
        paragraphs = paragraphs[:cfg["max_paragraphs"]]

    print(f"Loaded {len(paragraphs)} paragraphs from {cfg['input_file']} (format={used_fmt})")
    for i, p in enumerate(paragraphs[:max(0, int(cfg["print_first_n"]))]):
        preview = p.replace("\n", " ")[:120]
        print(f"[preview {i}] {preview}{'...' if len(p) > 120 else ''}")

    with ctx:
        results = compute_perplexity_over_paragraphs(model, paragraphs, enc, model.config.block_size, cfg["device"])

    print("----- Evaluation Results -----")
    print(f"model           : {cfg['init_from']}")
    print(f"paragraphs_used : {results['used_paragraphs']}")
    print(f"paragraphs_skip : {results['skipped_short']}")
    print(f"pred_tokens     : {results['pred_tokens']}")
    print(f"avg_loss        : {results['avg_loss']:.3f}")
    print(f"ppl             : {results['ppl']:.2f}")

    # sample generation, stopping at the second EOT to get a single clean story
    with ctx:
        sample = generate_text(
            model, cfg["prompt"], enc, cfg["device"],
            max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"], top_k=cfg["top_k"],
            eot_token=eot_id, stop_at_second_eot=True,
        )
    print("\n----- Sample Story -----")
    print(sample)


if __name__ == "__main__":
    main()
