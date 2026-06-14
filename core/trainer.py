"""
core/trainer.py

Shared training loop used by both the stories and code pipelines.

This module owns:
  - Device / DDP / dtype setup
  - The cosine-with-warmup learning-rate scheduler
  - The gradient-accumulation + AMP forward/backward step
  - Checkpoint save/resume logic
  - WandB logging (optional)
  - The outer training loop

Each pipeline calls `train()` with its own `get_batch` callable and model,
so the loop itself is completely task-agnostic.

Typical usage (from models/stories/train.py or models/code/train.py):

    from core.trainer import TrainConfig, train
    from core.data_utils import story_get_batch
    import functools

    cfg = TrainConfig(...)
    get_batch = functools.partial(story_get_batch, data_dir=..., ...)
    train(cfg, get_batch)
"""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# ---------------------------------------------------------------------------
# Training configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyper-parameters and I/O settings for a single training run.

    Fields map 1-to-1 with the old flat globals in train.py / train_for_task2.py
    but are now namespaced and type-annotated.
    """

    # -- Model architecture (passed through to GPTConfig) --
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = False

    # -- Data --
    dataset: str = "openwebtext"
    data_dir: str = ""          # resolved at runtime if left empty
    batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # -- I/O --
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: Literal["scratch", "resume"] = "scratch"

    # -- WandB --
    wandb_log: bool = False
    wandb_project: str = "nanogpt"
    wandb_run_name: str = "run"

    # -- Optimizer --
    learning_rate: float = 6e-4
    max_iters: int = 600_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # -- LR schedule --
    decay_lr: bool = True
    warmup_iters: int = 2_000
    lr_decay_iters: int = 600_000
    min_lr: float = 6e-5

    # -- System --
    backend: str = "nccl"
    device: str = "cuda"
    dtype: str = ""             # auto-detected if empty
    compile: bool = True

    def __post_init__(self) -> None:
        # Resolve data_dir from dataset name if not set explicitly.
        if not self.data_dir:
            self.data_dir = os.path.join("data", self.dataset)

        # Auto-detect best dtype if not provided.
        if not self.dtype:
            self.dtype = (
                "bfloat16"
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else "float16"
            )


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def _get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warm-up.

    Three regions:
      1. Linear warm-up from 0 → learning_rate over warmup_iters steps.
      2. Cosine decay from learning_rate → min_lr over lr_decay_iters steps.
      3. Constant min_lr after lr_decay_iters.
    """
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# Loss estimation (no-grad, many batches)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _estimate_loss(
    model: nn.Module,
    get_batch: Callable,
    cfg: TrainConfig,
    ctx,
) -> dict[str, float]:
    """Estimate mean loss over *eval_iters* batches for each split.

    Args:
        model:     The (possibly DDP-wrapped) model.
        get_batch: Callable(split) -> (x, y) tensors.
        cfg:       Training config.
        ctx:       AMP autocast context.

    Returns:
        Dict mapping "train" and "val" to their mean loss values.
    """
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _setup_ddp(cfg: TrainConfig) -> tuple[bool, bool, int, int, int]:
    """Initialise DDP if the RANK env var is set; return ddp metadata.

    Returns:
        (is_ddp, is_master, ddp_rank, ddp_local_rank, ddp_world_size)
    """
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        init_process_group(backend=cfg.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        is_master = ddp_rank == 0
        return is_ddp, is_master, ddp_rank, ddp_local_rank, ddp_world_size
    return False, True, 0, 0, 1


# ---------------------------------------------------------------------------
# Main train() entry-point
# ---------------------------------------------------------------------------

def train(
    cfg: TrainConfig,
    model: nn.Module,
    get_batch: Callable[[Literal["train", "val"]], tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Run the full training loop.

    Args:
        cfg:       Fully-resolved TrainConfig.
        model:     Uninitialised GPT model (already constructed, not yet on device).
        get_batch: Callable(split) -> (x, y) that returns one batch of data.
                   Typically a functools.partial of story_get_batch or
                   code_get_batch with data_dir, block_size, etc. already bound.
    """
    # ------------------------------------------------------------------ #
    # 1. DDP / device / seed setup                                        #
    # ------------------------------------------------------------------ #
    is_ddp, is_master, ddp_rank, ddp_local_rank, ddp_world_size = _setup_ddp(cfg)

    if is_ddp:
        cfg.device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(cfg.device)
        seed_offset = ddp_rank
        assert cfg.gradient_accumulation_steps % ddp_world_size == 0
        cfg.gradient_accumulation_steps //= ddp_world_size
    else:
        seed_offset = 0

    tokens_per_iter = (
        cfg.gradient_accumulation_steps * ddp_world_size * cfg.batch_size * cfg.block_size
    )
    print(f"tokens per iteration: {tokens_per_iter:,}")

    if is_master:
        os.makedirs(cfg.out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in cfg.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        cfg.dtype
    ]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # ------------------------------------------------------------------ #
    # 2. Model init / resume                                              #
    # ------------------------------------------------------------------ #
    iter_num = 0
    best_val_loss = 1e9

    if cfg.init_from == "resume":
        print(f"Resuming training from {cfg.out_dir}")
        ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)

        # Restore iteration counter and best loss from the checkpoint.
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

        # Load model weights (strip torch.compile prefix if present).
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    else:
        checkpoint = None

    model.to(cfg.device)

    # ------------------------------------------------------------------ #
    # 3. Scaler + optimizer                                               #
    # ------------------------------------------------------------------ #
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "float16"))
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type
    )
    if cfg.init_from == "resume" and checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free RAM

    # ------------------------------------------------------------------ #
    # 4. Optional torch.compile + DDP wrap                                #
    # ------------------------------------------------------------------ #
    if cfg.compile:
        print("Compiling model (takes ~1 min)…")
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model: nn.Module = model.module if is_ddp else model  # for checkpoint saves

    # ------------------------------------------------------------------ #
    # 5. WandB                                                            #
    # ------------------------------------------------------------------ #
    if cfg.wandb_log and is_master:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg))

    # ------------------------------------------------------------------ #
    # 6. Training loop                                                    #
    # ------------------------------------------------------------------ #
    X, Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    while True:
        # --- LR update ---
        lr = _get_lr(iter_num, cfg) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Eval + checkpoint ---
        if iter_num % cfg.eval_interval == 0 and is_master:
            losses = _estimate_loss(model, get_batch, cfg, ctx)
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            if cfg.wandb_log:
                import wandb
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    }
                )
            if losses["val"] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    ckpt = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": vars(cfg),
                    }
                    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
                    print(f"Saving checkpoint to {ckpt_path}")
                    torch.save(ckpt, ckpt_path)

        if iter_num == 0 and cfg.eval_only:
            break

        # --- Forward / backward with gradient accumulation ---
        for micro_step in range(cfg.gradient_accumulation_steps):
            if is_ddp:
                # Sync gradients only on the last micro-step.
                model.require_backward_grad_sync = (
                    micro_step == cfg.gradient_accumulation_steps - 1
                )
            with ctx:
                _, loss = model(X, Y)
                loss = loss / cfg.gradient_accumulation_steps

            # Prefetch next batch while the GPU runs backward.
            X, Y = get_batch("train")
            scaler.scale(loss).backward()

        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # --- Timing + logging ---
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0 and is_master:
            lossf = loss.item() * cfg.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    cfg.batch_size * cfg.gradient_accumulation_steps, dt
                )
                running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"iter {iter_num}: loss {lossf:.4f}, "
                f"time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
            )

        iter_num += 1
        local_iter_num += 1

        if iter_num > cfg.max_iters:
            break

    if is_ddp:
        destroy_process_group()