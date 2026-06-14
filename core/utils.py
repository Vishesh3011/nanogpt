"""
core/utils.py

Miscellaneous helpers shared across both pipelines.

Includes:
  - set_seed()           : deterministic seeding for reproducibility
  - get_device_type()    : "cuda" | "cpu" from a device string
  - load_checkpoint()    : load a ckpt.pt and return (model, state_dict, meta)
  - parse_config_file()  : replaces the old exec(open('configurator.py')) pattern
                           with an explicit, importable function
"""

from __future__ import annotations

import os
import sys
from ast import literal_eval
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int, device: str = "cpu") -> None:
    """Set random seeds for reproducible training.

    Args:
        seed:   Base integer seed.
        device: If a CUDA device is specified the CUDA seed is also set.
    """
    torch.manual_seed(seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device_type(device: str) -> str:
    """Return "cuda" or "cpu" from a full device string like "cuda:0".

    Used to decide whether to enable pinned memory, AMP, etc.
    """
    return "cuda" if "cuda" in device else "cpu"


def best_dtype(device_type: str) -> str:
    """Return the best floating-point dtype for the given device type.

    Prefers bfloat16 on CUDA if supported (avoids loss scaling), falls back
    to float16 for older GPUs, and float32 for CPU.
    """
    if device_type == "cuda":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    return "float32"


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint saved by core/trainer.py into *model* (and optionally *optimizer*).

    Handles the '_orig_mod.' prefix that torch.compile adds to state-dict keys.

    Args:
        ckpt_path:  Path to the .pt checkpoint file.
        model:      An already-instantiated model (same architecture as when saved).
        device:     Device to load tensors onto.
        optimizer:  If provided, optimizer state is restored from the checkpoint.

    Returns:
        The raw checkpoint dict (contains 'config', 'iter_num', 'best_val_loss', etc.)
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint["model"]
    # torch.compile wraps parameter names with '_orig_mod.'; strip it.
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    """Save a training checkpoint to *path*.

    Args:
        path:           Full file path (e.g. checkpoints/stories/ckpt.pt).
        model:          Raw (non-DDP-wrapped) model.
        optimizer:      Optimizer whose state should be saved.
        iter_num:       Current training iteration.
        best_val_loss:  Best validation loss seen so far.
        config:         Serialisable dict of training config values.
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Config parsing (replaces the old configurator.py exec() pattern)
# ---------------------------------------------------------------------------

def apply_config_overrides(namespace: dict[str, Any]) -> None:
    """Apply command-line and config-file overrides to *namespace* in-place.

    This is a clean replacement for the old ``exec(open('configurator.py').read())``
    pattern.  It reads ``sys.argv[1:]`` and:
      - Treats bare arguments (no ``=``) as config file paths and execs them.
      - Treats ``--key=value`` arguments as direct overrides with type-checking.

    Args:
        namespace: The dict of variables to mutate (typically ``globals()`` of
                   the calling script, or a dataclass converted to dict).

    Raises:
        ValueError: If an unknown config key is passed via CLI.
        AssertionError: If the value type does not match the existing type.
    """
    for arg in sys.argv[1:]:
        if "=" not in arg:
            # Bare arg → treat as a config file path.
            assert not arg.startswith("--"), (
                f"Expected a config file path or --key=value, got: {arg!r}"
            )
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                src = f.read()
            print(src)
            exec(compile(src, config_file, "exec"), namespace)  # noqa: S102
        else:
            # --key=value argument.
            assert arg.startswith("--"), f"Expected --key=value, got: {arg!r}"
            key, val = arg.split("=", 1)
            key = key[2:]
            if key not in namespace:
                raise ValueError(f"Unknown config key: {key!r}")
            try:
                typed_val = literal_eval(val)
            except (SyntaxError, ValueError):
                typed_val = val
            assert type(typed_val) is type(namespace[key]), (
                f"Type mismatch for key {key!r}: "
                f"expected {type(namespace[key])}, got {type(typed_val)}"
            )
            print(f"Overriding: {key} = {typed_val!r}")
            namespace[key] = typed_val
