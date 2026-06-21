"""
Shared data loading utilities.

Both pipelines store tokenized data as flat uint16 binary files (`train.bin`,
`val.bin`) under `data/<task>/`. This module provides memory-mapped batch
sampling, in two flavors:

  - `get_batch`: uniform random sampling of (input, target) windows (used by
    the code-generation pipeline).
  - `get_batch_aligned`: sampling windows that start at sequence boundaries
    (marked by the GPT-2 end-of-text token), used by the stories pipeline so
    that training examples tend to start at the beginning of a story.
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch

EOT_TOKEN_ID = 50256  # GPT-2 "<|endoftext|>" token id


def load_memmap(data_dir: str, split: str) -> np.memmap:
    """Memory-map the train or val token file for a given dataset directory.

    A fresh memmap is created on each call (rather than cached) to avoid the
    memory-leak issue described in:
    https://stackoverflow.com/questions/45132940
    """
    filename = "train.bin" if split == "train" else "val.bin"
    path = os.path.join(data_dir, filename)
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data_dir: str,
    split: str,
    batch_size: int,
    block_size: int,
    device: str,
    device_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, target) sequences via uniform random offsets."""
    data = load_memmap(data_dir, split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    return _to_device(x, y, device, device_type)


def _sequence_start_positions(data_dir: str, split: str, eot_token_id: int = EOT_TOKEN_ID) -> np.ndarray:
    """Return token offsets where a new sequence begins (position 0 and right after each EOT)."""
    data = load_memmap(data_dir, split)
    eot_pos = np.where(data == eot_token_id)[0]
    return np.concatenate([[0], eot_pos + 1])


def get_batch_aligned(
    data_dir: str,
    split: str,
    batch_size: int,
    block_size: int,
    device: str,
    device_type: str,
    start_positions_cache: Dict[str, np.ndarray],
    eot_token_id: int = EOT_TOKEN_ID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, target) sequences starting at sequence boundaries.

    `start_positions_cache` is a dict used to memoize the boundary positions per
    split across calls (pass the same dict on every call within a run).
    """
    data = load_memmap(data_dir, split)

    if split not in start_positions_cache:
        start_positions_cache[split] = _sequence_start_positions(data_dir, split, eot_token_id)
    starts = start_positions_cache[split]
    starts = starts[starts < len(data) - block_size]

    idx = np.random.randint(0, len(starts), size=batch_size)
    ix = starts[idx]

    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    return _to_device(x, y, device, device_type)


def _to_device(x: torch.Tensor, y: torch.Tensor, device: str, device_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
