"""
tokenizer.py — Stage 2: Tokenization.

Simple whitespace + punctuation tokenizer that preserves important symbols.
Optional tiktoken wrapper for accurate GPT token counting.
"""

import re
from typing import List, Optional


# Symbols that carry semantic meaning and must be preserved
PRESERVED_SYMBOLS = {'@', '#', '$', '%', '&', '*', '+', '=', '<', '>', '/',
                     '\\', '|', '~', '^', '{', '}', '[', ']', '(', ')'}


def tokenize(text: str) -> List[str]:
    """
    Tokenize text by splitting on whitespace and separating punctuation,
    while preserving important symbols as their own tokens.

    Returns:
        List of tokens.
    """
    tokens: List[str] = []
    # Split by whitespace first
    raw_parts = text.split()

    for part in raw_parts:
        tokens.extend(_split_token(part))

    return tokens


def detokenize(tokens: List[str]) -> str:
    """
    Reconstruct text from tokens with smart spacing.
    Handles punctuation attachment (no space before '.', ',', etc.).
    """
    if not tokens:
        return ""

    result = [tokens[0]]
    punct_attach = {'.', ',', '!', '?', ';', ':', "'", '"', ')', ']', '}'}
    no_space_after = {'(', '[', '{', '"', "'"}

    for token in tokens[1:]:
        if token in punct_attach:
            result.append(token)
        elif result and result[-1] in no_space_after:
            result.append(token)
        else:
            result.append(' ')
            result.append(token)

    return ''.join(result)


def count_tokens_tiktoken(text: str, model: str = "gpt-4") -> Optional[int]:
    """
    Count tokens using tiktoken for accurate cost estimation.
    Returns None if tiktoken is not available.
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except (ImportError, Exception):
        return None


def count_tokens_simple(text: str) -> int:
    """Simple token count approximation (whitespace split)."""
    return len(text.split())


# ── Internal ─────────────────────────────────────────────────────────────────

def _split_token(part: str) -> List[str]:
    """
    Split a single whitespace-delimited part into sub-tokens.
    Separates punctuation from words while preserving symbols.
    """
    tokens: List[str] = []

    # Handle pure preserved symbols
    if part in PRESERVED_SYMBOLS:
        return [part]

    # Regex: split into word chars, preserved symbols, or other punctuation
    pattern = r"(\w+|[" + re.escape(''.join(PRESERVED_SYMBOLS)) + r"]|[^\w\s])"
    matches = re.findall(pattern, part)

    return matches if matches else [part]
