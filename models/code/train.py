"""
Train the code generation model.

Usage:
    python -m models.code.train
    python -m models.code.train configs/code/train_config.py
    python -m models.code.train --max_iters=8000 --batch_size=64
"""

import os
import pickle

from core.data_utils import get_batch
from core.trainer import run_training
from core.utils import default_dtype, load_config
from models.code.model import default_model_args

DATA_DIR = os.path.join("data", "code")

DEFAULT_CONFIG = {
    # I/O
    "out_dir": os.path.join("checkpoints", "code"),
    "eval_interval": 200,
    "log_interval": 10,
    "eval_iters": 200,
    "eval_only": False,
    "always_save_checkpoint": False,
    "init_from": "scratch",  # 'scratch' | 'resume' | 'gpt2*'

    # wandb
    "wandb_log": False,
    "wandb_project": "nanogpt-code",
    "wandb_run_name": "code",

    # data / batching
    "gradient_accumulation_steps": 2,
    "batch_size": 64,
    "block_size": 128,

    # optimizer
    "learning_rate": 2e-4,
    "max_iters": 8000,
    "weight_decay": 0.05,
    "beta1": 0.9,
    "beta2": 0.99,
    "grad_clip": 1.0,

    # LR schedule
    "decay_lr": True,
    "warmup_iters": 200,
    "lr_decay_iters": 8000,
    "min_lr": 2e-5,

    # DDP
    "backend": "nccl",

    # system
    "device": "cuda",
    "dtype": default_dtype(),
    "compile": False,
    "seed": 1337,
}


def main() -> None:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(default_model_args())
    cfg = load_config(cfg)

    meta_path = os.path.join(DATA_DIR, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    def get_batch_fn(split: str):
        return get_batch(
            data_dir=DATA_DIR,
            split=split,
            batch_size=cfg["batch_size"],
            block_size=cfg["block_size"],
            device=cfg["device"],
            device_type="cuda" if "cuda" in cfg["device"] else "cpu",
        )

    run_training(cfg, get_batch_fn, meta_vocab_size=meta_vocab_size)


if __name__ == "__main__":
    main()
