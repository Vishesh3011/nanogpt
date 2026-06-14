"""
models/code/train.py

Training entrypoint for the Go code-generation model.

Mirrors models/stories/train.py but wires in the code-specific model
(RMSNorm + SwiGLU) and the random-window get_batch (no EOT boundary logic).

Usage (single GPU):
    python -m models.code.train configs/code/train_config.py

Usage (DDP, 4 GPUs):
    torchrun --standalone --nproc_per_node=4 -m models.code.train configs/code/train_config.py

Override values via CLI:
    python -m models.code.train configs/code/train_config.py --learning_rate=2e-4
"""

import functools

# ---------------------------------------------------------------------------
# Default hyper-parameters for the code model.
# These match the best config from the assignment (31M, CodeSearchNet Go).
# ---------------------------------------------------------------------------

# I/O
out_dir: str = "checkpoints/code"
eval_interval: int = 200
log_interval: int = 10
eval_iters: int = 200
eval_only: bool = False
always_save_checkpoint: bool = False
init_from: str = "scratch"  # "scratch" | "resume"

# WandB
wandb_log: bool = False
wandb_project: str = "nanogpt-code"
wandb_run_name: str = "code"

# Data
dataset: str = "code_search_net"
data_dir: str = "data/code"

# Model architecture (same size as stories for fair comparison)
n_layer: int = 6
n_head: int = 7       # 7 heads × 50 head_dim = 350 embd (assignment-era values)
n_embd: int = 350
block_size: int = 128
dropout: float = 0.1
bias: bool = False

# Batch / accumulation
batch_size: int = 64
gradient_accumulation_steps: int = 2

# Optimizer
learning_rate: float = 2e-4
max_iters: int = 8_000
weight_decay: float = 0.05
beta1: float = 0.9
beta2: float = 0.99
grad_clip: float = 1.0

# LR schedule
decay_lr: bool = True
warmup_iters: int = 200
lr_decay_iters: int = 8_000
min_lr: float = 2e-5

# System
backend: str = "nccl"
device: str = "cuda"
dtype: str = "float16"
compile: bool = False

# ---------------------------------------------------------------------------
# Apply config-file / CLI overrides
# ---------------------------------------------------------------------------
from core.utils import apply_config_overrides  # noqa: E402
apply_config_overrides(globals())

# ---------------------------------------------------------------------------
# Build TrainConfig, model, and get_batch
# ---------------------------------------------------------------------------
from core.trainer import TrainConfig, train     # noqa: E402
from core.data_utils import code_get_batch      # noqa: E402
from models.code.model import GPT, GPTConfig    # noqa: E402

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

get_batch = functools.partial(
    code_get_batch,
    data_dir=cfg.data_dir,
    block_size=cfg.block_size,
    batch_size=cfg.batch_size,
    device=cfg.device,
    device_type="cuda" if "cuda" in cfg.device else "cpu",
)

train(cfg, model, get_batch)