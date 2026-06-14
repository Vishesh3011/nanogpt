"""
models/code/model.py

GPT architecture for the Go code-generation pipeline.

Architecture choices (matching the original model2.py, more modern than stories):
  - RMSNorm instead of LayerNorm  — faster, no bias, used in LLaMA/Mistral
  - SwiGLU instead of GELU MLP   — gated activation, used in PaLM/LLaMA
  - No bias on any Linear         — consistent with RMSNorm, slightly faster
  - Weight tying between token embedding and LM head
  - Flash Attention when PyTorch >= 2.0

References:
  1. RMSNorm: https://arxiv.org/abs/1910.07467
  2. SwiGLU:  https://arxiv.org/abs/2002.05202
  3. LLaMA:   https://arxiv.org/abs/2302.13971
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

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no mean-centering, no bias).

    Compared to LayerNorm, RMSNorm is ~10% faster and achieves comparable
    quality, making it the preferred choice in modern LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (identical to the stories model).

    Uses Flash Attention (PyTorch >= 2.0) when available.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: Flash Attention unavailable, using slow manual attention.")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
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
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (gated SILU activation).

    Computes: dropout(W3 · (silu(W1·x) ⊙ W2·x))

    The gating mechanism (W2·x) allows the network to selectively suppress
    or amplify activation values, empirically improving perplexity vs. GELU
    at similar parameter counts.

    Note: uses 3 linear layers instead of 2, but the output dimension of W1/W2
    can be reduced to keep total parameter count equivalent to a GELU MLP.
    Here we use 4× expansion for simplicity (matching the stories model).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)  # gate input
        self.w2 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)  # gate
        self.w3 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)  # output
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Element-wise product of silu(w1·x) and w2·x, then project down.
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    """One Transformer block: pre-RMSNorm attention + pre-RMSNorm SwiGLU MLP."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Hyper-parameters for the code-generation GPT.

    Note: bias=False is the default here (unlike the stories model) because
    RMSNorm has no bias and the SwiGLU layers are also bias-free.
    """
    block_size: int = 1024
    vocab_size: int = 50304   # GPT-2 vocab padded to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False        # no bias (consistent with RMSNorm + SwiGLU)


# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """GPT language model for Go code generation.

    Uses the modern RMSNorm + SwiGLU stack. Otherwise structurally identical
    to the stories model — same training loop, same checkpoint format.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: token embedding and LM head share weights.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"GPT (code) initialised with {self.get_num_params() / 1e6:.2f}M parameters")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of trainable parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            idx:     Token indices, shape (B, T).
            targets: Optional shifted targets, shape (B, T).

        Returns:
            (logits, loss) — loss is None when targets is None.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Sequence length {t} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
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
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """Shrink positional embeddings for smaller-context inference."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        override_args: dict | None = None,
    ) -> "GPT":
        """Load GPT-2 weights from HuggingFace (for fine-tuning or eval baselines)."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)

        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained GPT-2: {model_type}")
        config_args: dict = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
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

        transposed = [
            "attn.c_attn.weight", "attn.c_proj.weight",
            "mlp.c_fc.weight",    "mlp.c_proj.weight",
        ]
        assert len(sd_keys_hf) == len(sd_keys)
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

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """Build AdamW with separate weight-decay groups and optional fused kernel."""
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

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate MFU as a fraction of A100 bfloat16 peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter
        return (flops_per_iter / dt) / 312e12

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Auto-regressively generate code tokens.

        Simpler than the stories generator — no EOT-boundary logic since
        code is a continuous stream without story delimiters.

        Args:
            idx:            Conditioning token indices, shape (B, T).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature:    Sampling temperature.
            top_k:          Top-k truncation (None = no truncation).

        Returns:
            Token indices of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
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

        return idx