"""
models/stories/model.py

GPT architecture for the story-generation pipeline.

Architecture choices (matching the original assignment model.py):
  - LayerNorm with optional bias  (not RMSNorm)
  - GELU activation in MLP        (not SwiGLU)
  - Weight tying between token embedding and LM head
  - Flash Attention when PyTorch >= 2.0

This is intentionally kept close to the original nanoGPT design so that
the assignment checkpoints remain compatible.  The code model (models/code/)
uses a more modern stack (RMSNorm + SwiGLU).

References:
  1. Official GPT-2 TF implementation:
     https://github.com/openai/gpt-2/blob/master/src/model.py
  2. HuggingFace GPT-2 PyTorch port:
     https://github.com/huggingface/transformers/blob/main/src/transformers/
     models/gpt2/modeling_gpt2.py
"""

from __future__ import annotations

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Layer normalisation with an optional bias parameter.

    PyTorch's built-in nn.LayerNorm always includes a bias; this wrapper
    lets us disable it (bias=False is slightly faster and sometimes better).
    """

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Uses Flash Attention (torch.nn.functional.scaled_dot_product_attention)
    when available (PyTorch >= 2.0); falls back to a manual implementation
    with a pre-registered causal mask otherwise.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            "Embedding dimension must be divisible by number of heads."
        )

        # Fused QKV projection: projects n_embd → 3 * n_embd in one linear.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection back to n_embd.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash Attention is ~3× faster but requires PyTorch >= 2.0.
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: Flash Attention unavailable, using slow manual attention.")
            # Causal mask stored as a buffer (not a parameter, not saved in state_dict).
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Split fused QKV into separate tensors and reshape for multi-head.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual causal attention.
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Merge heads and apply output projection.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: Linear → GELU → Linear → Dropout.

    The hidden dimension is 4× the embedding dimension, following the
    original GPT-2 design.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One Transformer block: pre-norm attention + pre-norm MLP with residuals."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Hyper-parameters that define the GPT architecture.

    Defaults match GPT-2 (124M).  The training configs in configs/stories/
    override these to a smaller ~31M model for the assignment constraint.
    """
    block_size: int = 1024
    # GPT-2 vocab is 50257; padded to nearest multiple of 64 for CUDA efficiency.
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # False is slightly faster and often better


# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """GPT language model for story generation.

    Combines token + positional embeddings, a stack of Transformer blocks,
    a final LayerNorm, and a linear LM head with weight tying.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),   # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),      # final layer norm
            )
        )
        # LM head projects hidden states → vocab logits.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share token embedding and LM head weights.
        # This saves ~38M parameters on a 768-dim model and improves training.
        # See: https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialise all weights.
        self.apply(self._init_weights)
        # Apply GPT-2's scaled init to residual projection weights.
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"GPT (stories) initialised with {self.get_num_params() / 1e6:.2f}M parameters")

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of trainable parameters.

        Args:
            non_embedding: If True, subtract position-embedding weights
                           (they are not used in the LM head, unlike token embeddings).
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise Linear weights with N(0, 0.02) and zero-init biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the model.

        Args:
            idx:     Integer token indices of shape (B, T).
            targets: Optional target token indices of shape (B, T).
                     If provided, the cross-entropy loss is computed and returned.

        Returns:
            (logits, loss) where loss is None when targets is None.
            During inference (no targets), logits has shape (B, 1, vocab_size)
            — only the last position is computed for efficiency.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Sequence length {t} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference optimisation: only project the last position.
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # ------------------------------------------------------------------
    # Model surgery
    # ------------------------------------------------------------------

    def crop_block_size(self, block_size: int) -> None:
        """Shrink the positional embedding table to a smaller block size.

        Useful when loading a checkpoint trained with a large block_size but
        running inference or fine-tuning with a smaller context.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # ------------------------------------------------------------------
    # Pre-trained weight loading (GPT-2 from HuggingFace)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        override_args: dict | None = None,
    ) -> "GPT":
        """Load GPT-2 weights from HuggingFace Transformers.

        Args:
            model_type:    One of "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl".
            override_args: Optional dict — only "dropout" can be overridden.

        Returns:
            A GPT instance with pre-trained weights.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args), (
            "Only 'dropout' can be overridden in from_pretrained()"
        )

        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained GPT-2: {model_type}")

        config_args: dict = {
            "gpt2":         dict(n_layer=12, n_head=12, n_embd=768),   # 124M
            "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args.update(vocab_size=50257, block_size=1024, bias=True)
        if "dropout" in override_args:
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [
            k for k in sd_hf
            if not k.endswith((".attn.masked_bias", ".attn.bias"))
        ]

        # HuggingFace uses Conv1D weights that need to be transposed.
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        assert len(sd_keys_hf) == len(sd_keys), (
            f"Key count mismatch: {len(sd_keys_hf)} HF keys vs {len(sd_keys)} model keys"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # ------------------------------------------------------------------
    # Optimizer configuration
    # ------------------------------------------------------------------

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """Build an AdamW optimizer with separate weight-decay groups.

        2D parameters (weight matrices, embeddings) are decayed;
        1D parameters (biases, LayerNorm scales) are not.

        Uses the fused AdamW kernel on CUDA for speed.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Decayed tensors: {len(decay_params)} ({n_decay:,} params)")
        print(f"Non-decayed tensors: {len(nodecay_params)} ({n_nodecay:,} params)")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra = {"fused": True} if use_fused else {}
        print(f"Using fused AdamW: {use_fused}")

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)

    # ------------------------------------------------------------------
    # MFU estimation
    # ------------------------------------------------------------------

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate Model FLOP Utilisation (MFU) as a fraction of A100 peak.

        Formula from PaLM Appendix B (https://arxiv.org/abs/2204.02311).

        Args:
            fwdbwd_per_iter: Number of tokens processed (batch × grad_accum).
            dt:              Wall-clock seconds per iteration.

        Returns:
            MFU as a float in [0, 1].
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12  # A100 bfloat16 peak FLOPS
        return flops_achieved / flops_promised

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eot_token: int = 50256,
    ) -> torch.Tensor:
        """Auto-regressively generate tokens, with story-boundary awareness.

        For short prompts (≤ 2 tokens), the model is in "new story" mode: it
        generates until it sees two consecutive <|endoftext|> tokens, then
        returns only the tokens between them (i.e. a single clean story).

        For longer prompts, it generates until the first <|endoftext|> or
        max_new_tokens is reached.

        Args:
            idx:            Conditioning token indices, shape (B, T).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature:    Sampling temperature (< 1 = sharper, > 1 = flatter).
            top_k:          If set, restrict sampling to the top-k logits.
            eot_token:      The <|endoftext|> token id (50256 for GPT-2).

        Returns:
            Token indices of the generated story (or full sequence if no EOT).
        """
        is_new_story = idx.size(1) <= 2
        eot_positions: list[int] = []

        for _ in range(max_new_tokens):
            # Crop context to block_size if needed.
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if idx_next.item() == eot_token:
                eot_positions.append(idx.size(1) - 1)
                if is_new_story:
                    if len(eot_positions) == 2:
                        break
                else:
                    break

        # In new-story mode, return only the tokens between the two EOTs.
        if is_new_story and len(eot_positions) >= 2:
            return idx[:, eot_positions[0] + 1 : eot_positions[1]]

        return idx