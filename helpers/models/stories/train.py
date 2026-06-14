"""
models/stories/train.py

Training entrypoint for the story-generation model.

This script wires together the shared core/ library with the stories-specific
model and data pipeline.  All heavy logic lives in core/; this file is just
configuration + wiring.

Usage (single GPU):
    python -m models.stories.train configs/stories/train_config.py

Usage (DDP, 4 GPUs):
    torchrun --standalone --nproc_per_node=4 -m models.stories.train configs/stories/train_config.py

Override individual values via CLI:
    python -m models.stories.train configs/stories/train_config.py --learning_rate=3e-4
"""

import functools
import os
import sys

# ---------------------------------------------------------------------------
# Default hyper-parameters
# These match the best assignment config (31M, ROCStories).
# Override via a config file or --key=value CLI args (see core/utils.py).
# ---------------------------------------------------------------------------

# I/O
out_dir: str = "checkpoints/stories"
eval_interval: int = 200
log_interval: int = 10
eval_iters: int = 200
eval_only: bool = False
always_save_checkpoint: bool = False
init_from: str = "scratch"  # "scratch" | "resume"

# WandB
wandb_log: bool = False
wandb_project: str = "nanogpt-stories"
wandb_run_name: str = "stories"

# Data
dataset: str = "tinystories"     # switched from rocstories for the rebuild
data_dir: str = "data/stories"

# Model architecture
n_layer: int = 8
n_head: int = 8
n_embd: int = 512
block_size: int = 256
dropout: float = 0.1
bias: bool = False

# Batch / accumulation
batch_size: int = 64
gradient_accumulation_steps: int = 2

# Optimizer
learning_rate: float = 3e-4
max_iters: int = 20_000
weight_decay: float = 0.1
beta1: float = 0.9
beta2: float = 0.95
grad_clip: float = 1.0

# LR schedule
decay_lr: bool = True
warmup_iters: int = 500
lr_decay_iters: int = 20_000
min_lr: float = 3e-5

# System
backend: str = "nccl"
device: str = "cuda"
dtype: str = ""      # auto-detected
compile: bool = False

# ---------------------------------------------------------------------------
# Apply config-file / CLI overrides (replaces old configurator.py exec trick)
# ---------------------------------------------------------------------------
from core.utils import apply_config_overrides  # noqa: E402
apply_config_overrides(globals())

# ---------------------------------------------------------------------------
# Build TrainConfig, model, and get_batch, then hand off to core/trainer.py
# ---------------------------------------------------------------------------
import functools  # noqa: E402 (already imported above, kept for clarity)

from core.trainer import TrainConfig, train          # noqa: E402
from core.data_utils import story_get_batch          # noqa: E402
from models.stories.model import GPT, GPTConfig      # noqa: E402

# Assemble TrainConfig from the (possibly overridden) globals.
cfg = TrainConfig(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    dropout=dropout,
    bias=bias,
    dataset=dataset,
    data_dir=data_dir,
    batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    out_dir=out_dir,
    eval_interval=eval_interval,
    log_interval=log_interval,
    eval_iters=eval_iters,
    eval_only=eval_only,
    always_save_checkpoint=always_save_checkpoint,
    init_from=init_from,
    wandb_log=wandb_log,
    wandb_project=wandb_project,
    wandb_run_name=wandb_run_name,
    learning_rate=learning_rate,
    max_iters=max_iters,
    weight_decay=weight_decay,
    beta1=beta1,
    beta2=beta2,
    grad_clip=grad_clip,
    decay_lr=decay_lr,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    min_lr=min_lr,
    backend=backend,
    device=device,
    dtype=dtype,
    compile=compile,
)

# Build the model.
gpt_config = GPTConfig(
    block_size=cfg.block_size,
    vocab_size=50304,
    n_layer=cfg.n_layer,
    n_head=cfg.n_head,
    n_embd=cfg.n_embd,
    dropout=cfg.dropout,
    bias=cfg.bias,
)
model = GPT(gpt_config)

# Bind dataset-specific args to the shared get_batch function.
get_batch = functools.partial(
    story_get_batch,
    data_dir=cfg.data_dir,
    block_size=cfg.block_size,
    batch_size=cfg.batch_size,
    device=cfg.device,
    device_type="cuda" if "cuda" in cfg.device else "cpu",
)

# Hand off to the shared training loop.
train(cfg, model, get_batch)