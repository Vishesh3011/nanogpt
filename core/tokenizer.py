"""
core/tokenizer.py

Thin wrapper around tiktoken's GPT-2 BPE tokenizer.
Both the stories and code pipelines share this tokenizer.

Usage:
    from core.tokenizer import Tokenizer
    tok = Tokenizer()
    ids  = tok.encode("hello world")
    text = tok.decode(ids)
"""

from __future__ import annotations

import tiktoken


# Token id for GPT-2's <|endoftext|> special token.
EOT_TOKEN_ID: int = 50256


class Tokenizer:
    """GPT-2 BPE tokenizer backed by tiktoken.

    This is intentionally a thin wrapper — it exposes only the encode/decode
    interface that the rest of the codebase needs, so swapping the underlying
    library later requires changing only this file.
    """

    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("gpt2")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (50257 for GPT-2)."""
        return self._enc.n_vocab

    @property
    def eot_token(self) -> int:
        """Integer id of the end-of-text special token."""
        return EOT_TOKEN_ID

    def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
        """Encode *text* into a list of integer token ids.

        Args:
            text:          The string to encode.
            allow_special: If True, the <|endoftext|> token is handled as a
                           special token rather than encoded literally.
        """
        allowed = {"<|endoftext|>"} if allow_special else set()
        return self._enc.encode(text, allowed_special=allowed)

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of integer token ids back to a string."""
        return self._enc.decode(token_ids)