"""
Model configuration preset for the "code" pipeline.

Architecture: LLaMA style
  - RMSNorm (no bias) + SwiGLU MLP
  - Grouped Query Attention: 2 KV heads shared across 8 Q heads
  - RoPE positional embeddings (no learned wpe)
  - GPT-2 BPE tokenizer, vocab_size=50304
  - <GO> / </GO> delimiters for function-level generation

Presets:
  default_model_args()  ~29M total params  (assignment-compatible)
  scaled_model_args()   ~49M total params  (Colab target)
"""

from core.architectures import GPTConfig


def default_model_args() -> dict:
    """~29M total param config (assignment-compatible)."""
    return dict(
        n_layer    = 6,
        n_head     = 7,
        n_kv_head  = 7,       # full MHA
        n_embd     = 350,
        block_size = 256,
        bias       = False,
        dropout    = 0.1,
        vocab_size = 50304,
        norm_type  = "rmsnorm",
        mlp_type   = "swiglu",
    )


def scaled_model_args() -> dict:
    """~49M total param config for Colab training on CodeSearchNet Go.

    Key improvements over default:
      - 12 layers vs 6 (much deeper — code benefits from depth more than stories)
      - n_embd 448 vs 350 (wider)
      - block_size 512 vs 256 (fits most Go functions in one window)
      - GQA with 2 KV heads: saves ~3M params in attention, spent on depth instead
      - RMSNorm + SwiGLU: best known combo for code generation
    """
    return dict(
        n_layer    = 12,
        n_head     = 8,
        n_kv_head  = 2,       # GQA: 4 Q heads share each KV head
        n_embd     = 448,
        block_size = 512,
        bias       = False,
        dropout    = 0.1,
        vocab_size = 50304,
        norm_type  = "rmsnorm",
        mlp_type   = "swiglu",
    )


def build_config(args: dict) -> GPTConfig:
    """Construct a GPTConfig from a model-args dict."""
    return GPTConfig(**args)
