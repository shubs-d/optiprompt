"""
text_utils.py — Shared text utilities for OptiPrompt.

Provides sentence splitting, word counting, n-gram generation,
stopword/filler sets, and other text helpers used across the pipeline.
"""

import re
from typing import List, Set, Tuple


# ── Stopwords (high-frequency, low-information words) ────────────────────────
from app.core.kb import kb

STOPWORDS: Set[str] = kb.stopwords

# ── Filler phrases (ordered longest-first for greedy matching) ───────────────
FILLER_PHRASES: List[str] = kb.filler_phrases

# ── Phrase replacement map ───────────────────────────────────────────────────
PHRASE_REPLACEMENTS: List[Tuple[str, str]] = kb.phrase_replacements

# Sort by length (longest first) for greedy replacement
PHRASE_REPLACEMENTS.sort(key=lambda x: len(x[0]), reverse=True)

# ── Low-value adjectives/adverbs ────────────────────────────────────────────
LOW_VALUE_MODIFIERS: Set[str] = kb.low_value_modifiers


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex-based rules."""
    # Split on period/exclamation/question followed by space+uppercase or end
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in parts if s.strip()]


def count_words(text: str) -> int:
    """Count words in text (whitespace-split)."""
    return len(text.split())


def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def is_meaningful_token(token: str) -> bool:
    """Check if a token carries semantic meaning (not a stopword or punct)."""
    cleaned = token.lower().strip(".,!?;:\"'()-")
    if not cleaned:
        return False
    if cleaned in STOPWORDS:
        return False
    if len(cleaned) <= 1 and not cleaned.isalpha():
        return False
    return True


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single spaces."""
    return re.sub(r'\s+', ' ', text).strip()
