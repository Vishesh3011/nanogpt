"""
Shared training loop.

Both the stories and code pipelines call `run_training(cfg, get_batch_fn)`
with a config dict and a batch-sampling function. This consolidates the
duplicated training loops from the original `train.py` / `train_for_task2.py`.

Distributed (DDP) training is supported but optional; single-GPU / CPU /
MPS (Apple Silicon) all work via the `device` config key.
"""

import math
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from core.architectures import GPT, GPTConfig
from core.utils import save_checkpoint, seed_everything, setup_device

# Type alias: a function (split: str) -> (x, y) tensors
BatchFn = Callable[[str], Tuple[torch.Tensor, torch.Tensor]]


def get_lr(it: int, cfg: Dict[str, Any]) -> float:
    """Cosine learning rate schedule with linear warmup."""
    warmup_iters = cfg["warmup_iters"]
    lr_decay_iters = cfg["lr_decay_iters"]
    learning_rate = cfg["learning_rate"]
    min_lr = cfg["min_lr"]

    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def setup_ddp(cfg: Dict[str, Any]) -> Tuple[bool, bool, int, Dict[str, Any]]:
    """Initialize DDP if running under torchrun.

    Returns:
        (is_ddp, is_master_process, ddp_local_rank, updated_cfg)
    """
    cfg = dict(cfg)
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        init_process_group(backend=cfg.get("backend", "nccl"))
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        cfg["device"] = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(cfg["device"])
        master_process = ddp_rank == 0
        cfg["seed_offset"] = ddp_rank
        cfg["ddp_world_size"] = ddp_world_size
        assert cfg["gradient_accumulation_steps"] % ddp_world_size == 0
        cfg["gradient_accumulation_steps"] //= ddp_world_size
        return True, master_process, ddp_local_rank, cfg

    cfg["seed_offset"] = 0
    return False, True, 0, cfg


def build_model(cfg: Dict[str, Any], meta_vocab_size: Optional[int]) -> Tuple[GPT, Dict[str, Any], int, float]:
    """Construct (or resume/load) the GPT model according to `init_from`.

    Returns:
        (model, model_args, iter_num, best_val_loss)
    """
    model_args = dict(
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_kv_head=cfg.get("n_kv_head", cfg["n_head"]),  # default: full MHA
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        bias=cfg["bias"],
        vocab_size=None,
        dropout=cfg["dropout"],
        norm_type=cfg.get("norm_type", "layernorm"),
        mlp_type=cfg.get("mlp_type", "gelu"),
    )

    iter_num = 0
    best_val_loss = 1e9
    init_from = cfg["init_from"]

    if init_from == "scratch":
        print("Initializing a new model from scratch")
        if meta_vocab_size is None:
            print("defaulting vocab_size to 50304 (GPT-2's 50257 rounded up for efficiency)")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        model = GPT(GPTConfig(**model_args))

    elif init_from == "resume":
        out_dir = cfg["out_dir"]
        print(f"Resuming training from {out_dir}")
        from core.utils import load_checkpoint
        checkpoint = load_checkpoint(os.path.join(out_dir, "ckpt.pt"), map_location=cfg["device"])
        checkpoint_model_args = checkpoint["model_args"]
        for k in ["n_layer", "n_head", "n_kv_head", "n_embd", "block_size", "bias", "vocab_size", "norm_type", "mlp_type"]:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        model = GPT(GPTConfig(**model_args))
        model.load_state_dict(checkpoint["model"])
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    elif init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        model = GPT.from_pretrained(init_from, dict(dropout=cfg["dropout"]))
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "norm_type", "mlp_type"]:
            model_args[k] = getattr(model.config, k)
        model_args["vocab_size"] = model.config.vocab_size

    else:
        raise ValueError(f"Unsupported init_from: {init_from}")

    if cfg["block_size"] < model.config.block_size:
        model.crop_block_size(cfg["block_size"])
        model_args["block_size"] = cfg["block_size"]

    return model, model_args, iter_num, best_val_loss


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, get_batch_fn: BatchFn, eval_iters: int, ctx: Any
) -> Dict[str, float]:
    """Estimate average train/val loss over `eval_iters` batches each."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def run_training(
    cfg: Dict[str, Any],
    get_batch_fn: BatchFn,
    meta_vocab_size: Optional[int] = None,
) -> None:
    """Run the full training loop given a config dict and batch-sampling function.

    Required keys in `cfg`: out_dir, eval_interval, log_interval, eval_iters,
    eval_only, always_save_checkpoint, init_from, gradient_accumulation_steps,
    batch_size, block_size, n_layer, n_head, n_embd, dropout, bias,
    learning_rate, max_iters, weight_decay, beta1, beta2, grad_clip, decay_lr,
    warmup_iters, lr_decay_iters, min_lr, backend, device, dtype, compile,
    wandb_log, wandb_project, wandb_run_name, seed.

    Optional: norm_type, mlp_type (default 'layernorm'/'gelu').
    """
    ddp, master_process, ddp_local_rank, cfg = setup_ddp(cfg)

    device = cfg["device"]
    device_type, ptdtype, ctx = setup_device(device, cfg["dtype"])

    tokens_per_iter = (
        cfg["gradient_accumulation_steps"] * cfg.get("ddp_world_size", 1)
        * cfg["batch_size"] * cfg["block_size"]
    )
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(cfg["out_dir"], exist_ok=True)
    seed_everything(cfg["seed"], cfg["seed_offset"])

    # model
    model, model_args, iter_num, best_val_loss = build_model(cfg, meta_vocab_size)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["dtype"] == "float16"))

    optimizer = model.configure_optimizers(
        cfg["weight_decay"], cfg["learning_rate"], (cfg["beta1"], cfg["beta2"]), device_type
    )
    if cfg["init_from"] == "resume":
        from core.utils import load_checkpoint
        checkpoint = load_checkpoint(os.path.join(cfg["out_dir"], "ckpt.pt"), map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        checkpoint = None

    if cfg["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    if cfg["wandb_log"] and master_process:
        import wandb
        wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)

    # training loop
    X, Y = get_batch_fn("train")
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    while True:
        lr = get_lr(iter_num, cfg) if cfg["decay_lr"] else cfg["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % cfg["eval_interval"] == 0 and master_process:
            losses = estimate_loss(model, get_batch_fn, cfg["eval_iters"], ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if cfg["wandb_log"]:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                })
            if losses["val"] < best_val_loss or cfg["always_save_checkpoint"]:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    print(f"saving checkpoint to {cfg['out_dir']}")
                    save_checkpoint(
                        os.path.join(cfg["out_dir"], "ckpt.pt"),
                        raw_model, optimizer, model_args, iter_num, best_val_loss, cfg,
                    )

        if iter_num == 0 and cfg["eval_only"]:
            break

        for micro_step in range(cfg["gradient_accumulation_steps"]):
            if ddp:
                model.require_backward_grad_sync = (micro_step == cfg["gradient_accumulation_steps"] - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg["gradient_accumulation_steps"]
            X, Y = get_batch_fn("train")
            scaler.scale(loss).backward()

        if cfg["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg["log_interval"] == 0 and master_process:
            lossf = loss.item() * cfg["gradient_accumulation_steps"]
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(cfg["batch_size"] * cfg["gradient_accumulation_steps"], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        if iter_num > cfg["max_iters"]:
            break

    if ddp:
        destroy_process_group()
