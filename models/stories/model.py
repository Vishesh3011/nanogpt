"""
Model configuration preset for the "stories" pipeline.

Architecture: GPT-2 style
  - LayerNorm + bias + GELU MLP
  - Full Multi-Head Attention (n_kv_head == n_head; GQA disabled)
  - RoPE positional embeddings (no learned wpe)
  - GPT-2 BPE tokenizer, vocab_size=50304
  - "<|endoftext|>" as story-boundary / EOT token

Presets:
  default_model_args()  ~31M total params  (assignment-compatible)
  scaled_model_args()   ~49M total params  (Colab / TinyStories target)
"""

from core.architectures import GPTConfig


def default_model_args() -> dict:
    """~31M total param config (assignment-compatible, original block_size preserved)."""
    return dict(
        n_layer    = 7,
        n_head     = 6,
        n_kv_head  = 6,       # full MHA
        n_embd     = 384,
        block_size = 256,
        bias       = True,
        dropout    = 0.1,
        vocab_size = 50304,
        norm_type  = "layernorm",
        mlp_type   = "gelu",
    )


def scaled_model_args() -> dict:
    """~49M total param config for Colab training on TinyStories.

    Key improvements over default:
      - 9 layers vs 7 (more depth)
      - n_embd 480 vs 384 (wider residual stream)
      - block_size 512 vs 256 (longer context for richer stories)
      - Full MHA (n_kv_head == n_head) — best for smaller datasets
      - RoPE handles position (no wpe cost)
    """
    return dict(
        n_layer    = 9,
        n_head     = 8,
        n_kv_head  = 8,       # full MHA — better for generalisation at this scale
        n_embd     = 480,
        block_size = 512,
        bias       = True,
        dropout    = 0.1,
        vocab_size = 50304,
        norm_type  = "layernorm",
        mlp_type   = "gelu",
    )


def build_config(args: dict) -> GPTConfig:
    """Construct a GPTConfig from a model-args dict."""
    return GPTConfig(**args)
