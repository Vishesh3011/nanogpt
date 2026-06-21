"""
Shared utilities: config loading/overriding, reproducibility seeding,
device/dtype setup, and checkpoint load/save helpers.

This replaces the original `configurator.py` "poor man's configurator"
(which used `exec()` on CLI args) with an explicit, importable function.
"""

import os
import sys
from ast import literal_eval
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Config loading / CLI overrides
# ---------------------------------------------------------------------------

def load_config(config: Dict[str, Any], argv: Optional[list] = None) -> Dict[str, Any]:
    """Apply CLI overrides to a base config dict.

    Supports two forms of arguments (mirroring the original configurator.py):
      - a bare path to a Python file, which is exec'd to override keys
      - `--key=value` pairs, where value is parsed with `ast.literal_eval`
        when possible (so ints/floats/bools/None parse correctly), falling
        back to a raw string otherwise.

    Args:
        config: base configuration dict (mutated copy is returned).
        argv: argument list to parse (defaults to sys.argv[1:]).

    Returns:
        A new dict with overrides applied.
    """
    cfg = dict(config)
    args = sys.argv[1:] if argv is None else argv

    for arg in args:
        if "=" not in arg:
            # treat as a path to a config file to exec for overrides
            assert not arg.startswith("--"), f"Unexpected flag without value: {arg}"
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                file_globals: Dict[str, Any] = {}
                exec(f.read(), file_globals)
            for k, v in file_globals.items():
                if k.startswith("_"):
                    continue
                cfg[k] = v
        else:
            assert arg.startswith("--"), f"Expected --key=value, got: {arg}"
            key, val = arg.split("=", 1)
            key = key[2:]
            if key in cfg:
                try:
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    attempt = val
                print(f"Overriding: {key} = {attempt}")
                cfg[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")

    return cfg


# ---------------------------------------------------------------------------
# Reproducibility / device setup
# ---------------------------------------------------------------------------

def seed_everything(seed: int, seed_offset: int = 0) -> None:
    """Seed torch (CPU + CUDA) for reproducibility."""
    torch.manual_seed(seed + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + seed_offset)


def setup_device(device: str, dtype: str) -> Tuple[str, torch.dtype, Any]:
    """Resolve device type, torch dtype, and an autocast context manager.

    Args:
        device: e.g. 'cpu', 'cuda', 'cuda:0', 'mps'
        dtype: one of 'float32', 'bfloat16', 'float16'

    Returns:
        (device_type, torch_dtype, autocast_context_manager)
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else ("mps" if "mps" in device else "cpu")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

    if device_type == "cpu":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    return device_type, ptdtype, ctx


def default_dtype() -> str:
    """Pick the best available dtype: bfloat16 if supported, else float16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the '_orig_mod.' prefix added by torch.compile() to state_dict keys."""
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_args: Dict[str, Any],
    iter_num: int,
    best_val_loss: float,
    config: Dict[str, Any],
) -> None:
    """Save a training checkpoint to `path`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dict and strip any torch.compile prefixes from the state dict."""
    checkpoint = torch.load(path, map_location=map_location)
    if "model" in checkpoint:
        checkpoint["model"] = strip_compile_prefix(checkpoint["model"])
    return checkpoint


def config_to_dict(config_obj: Any) -> Dict[str, Any]:
    """Convert a dataclass config (e.g. GPTConfig) to a plain dict."""
    return asdict(config_obj)
