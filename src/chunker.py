"""
chunker.py
Splits long contracts into overlapping chunks that fit within the model's
effective context budget, so extraction doesn't silently truncate text.

Most CUAD contracts fit in a single call to a long-context model like Claude,
so chunking is the exception, not the rule -- but the pipeline should not
break on the handful of very long contracts (some CUAD documents run 80+ pages).
"""

from typing import List

import tiktoken

from config import MAX_CONTEXT_TOKENS, CHUNK_OVERLAP_TOKENS

# tiktoken doesn't ship a Claude-specific encoding; cl100k_base gives a close
# enough token-count estimate for chunk-sizing purposes.
_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def needs_chunking(text: str, max_tokens: int = MAX_CONTEXT_TOKENS) -> bool:
    return count_tokens(text) > max_tokens


def chunk_text(
    text: str,
    max_tokens: int = MAX_CONTEXT_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """
    Split text into overlapping chunks of at most `max_tokens` tokens.
    Overlap ensures a clause split across a chunk boundary still appears
    in full in at least one chunk.
    """
    tokens = _encoding.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_encoding.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - overlap_tokens  # step back to create overlap

    return chunks
