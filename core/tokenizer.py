"""
Tokenizer utilities shared by both the stories and code pipelines.

Both models use the GPT-2 BPE tokenizer (via tiktoken), with the
"<|endoftext|>" special token treated as an allowed token for
sequence/story/function boundaries.
"""

from typing import List

import tiktoken

EOT_TOKEN_STR = "<|endoftext|>"


def get_tokenizer() -> tiktoken.Encoding:
    """Return the shared GPT-2 BPE tokenizer."""
    return tiktoken.get_encoding("gpt2")


def get_eot_token_id(enc: tiktoken.Encoding) -> int:
    """Return the integer id of the end-of-text token for this tokenizer."""
    return enc.encode(EOT_TOKEN_STR, allowed_special={EOT_TOKEN_STR})[0]


def encode(text: str, enc: tiktoken.Encoding) -> List[int]:
    """Encode text to token ids, allowing the end-of-text special token."""
    return enc.encode(text, allowed_special={EOT_TOKEN_STR})


def decode(token_ids: List[int], enc: tiktoken.Encoding) -> str:
    """Decode token ids back to text."""
    return enc.decode(token_ids)
