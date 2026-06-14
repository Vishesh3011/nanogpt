"""
core/data_utils.py

Shared data-loading utilities used by both the stories and code pipelines.

Responsibilities:
  - Loading pre-tokenised binary shards (train.bin / val.bin) from disk via
    memory-mapped numpy arrays.
  - Providing task-specific get_batch implementations:
      * story_get_batch  — samples story-boundary-aligned windows (EOT-aware)
      * code_get_batch   — samples random windows (code has no story boundaries)
  - A lightweight EOT-position cache so boundary scanning only happens once
    per process lifetime.

Both get_batch variants return (x, y) tensors on the correct device.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torch

from core.tokenizer import EOT_TOKEN_ID


# ---------------------------------------------------------------------------
# Internal cache: maps (data_dir, split) -> array of story-start positions
# ---------------------------------------------------------------------------
_eot_cache: dict[tuple[str, str], np.ndarray] = {}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_shard(data_dir: str, split: Literal["train", "val"]) -> np.memmap:
    """Return a memory-mapped view of train.bin or val.bin."""
    filename = "train.bin" if split == "train" else "val.bin"
    path = os.path.join(data_dir, filename)
    return np.memmap(path, dtype=np.uint16, mode="r")


def _get_story_starts(data_dir: str, split: Literal["train", "val"]) -> np.ndarray:
    """Return (and cache) an array of token indices that start a new story.

    A story begins immediately after each <|endoftext|> token as well as at
    position 0.  This is pre-computed once per (data_dir, split) pair and
    stored in a module-level dict so repeated calls are O(1).
    """
    key = (data_dir, split)
    if key not in _eot_cache:
        data = _load_shard(data_dir, split)
        eot_positions = np.where(data == EOT_TOKEN_ID)[0]
        starts = np.concatenate([[0], eot_positions + 1])
        _eot_cache[key] = starts
    return _eot_cache[key]


def _to_device(
    x: torch.Tensor,
    y: torch.Tensor,
    device: str,
    device_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move (x, y) tensors to *device*, using pinned memory for CUDA."""
    if device_type == "cuda":
        return (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Public batch functions
# ---------------------------------------------------------------------------

def story_get_batch(
    split: Literal["train", "val"],
    *,
    data_dir: str,
    block_size: int,
    batch_size: int,
    device: str,
    device_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch aligned to story boundaries (for the stories pipeline).

    Starting positions are chosen from the set of token indices that
    immediately follow an <|endoftext|> token.  This keeps each training
    window anchored at the start of a story rather than in the middle of one,
    which materially improves generation quality on short-story datasets.

    Args:
        split:       "train" or "val".
        data_dir:    Path to the directory containing train.bin / val.bin.
        block_size:  Context length (number of tokens per window).
        batch_size:  Number of windows per batch.
        device:      PyTorch device string (e.g. "cuda", "cpu").
        device_type: "cuda" or "cpu" — controls whether pinned memory is used.

    Returns:
        (x, y) where x is the input window and y is x shifted by one token.
        Both have shape (batch_size, block_size) and dtype int64.
    """
    data = _load_shard(data_dir, split)
    story_starts = _get_story_starts(data_dir, split)

    # Only keep starts that leave room for a full block_size window.
    valid_starts = story_starts[story_starts < len(data) - block_size]

    idx = np.random.randint(0, len(valid_starts), size=batch_size)
    ix = valid_starts[idx]

    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )

    return _to_device(x, y, device, device_type)


def code_get_batch(
    split: Literal["train", "val"],
    *,
    data_dir: str,
    block_size: int,
    batch_size: int,
    device: str,
    device_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch at random positions (for the code pipeline).

    Code tokens form a continuous stream without meaningful story boundaries,
    so random sampling is appropriate here.

    Args:
        split:       "train" or "val".
        data_dir:    Path to the directory containing train.bin / val.bin.
        block_size:  Context length (number of tokens per window).
        batch_size:  Number of windows per batch.
        device:      PyTorch device string.
        device_type: "cuda" or "cpu".

    Returns:
        (x, y) where x is the input window and y is x shifted by one token.
        Both have shape (batch_size, block_size) and dtype int64.
    """
    data = _load_shard(data_dir, split)

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )

    return _to_device(x, y, device, device_type)