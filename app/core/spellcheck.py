"""
spellcheck.py — Post-optimization deterministic spell correction.

Uses difflib.get_close_matches against a controlled vocabulary to fix
common misspellings in optimized prompts. No ML, no external APIs.
"""

import re
from difflib import get_close_matches
from typing import List, Optional


# ── Controlled vocabulary for spell correction ───────────────────────────────
# Words commonly found in optimized prompts. Keep this list curated and small
# to avoid false-positive corrections.
VOCAB: List[str] = [
    # Actions
    "build", "create", "generate", "explain", "analyze", "design",
    "implement", "optimize", "develop", "deploy", "configure", "integrate",
    "test", "debug", "refactor", "document", "describe", "summarize",
    "write", "read", "update", "delete", "list", "search", "filter",
    # Nouns
    "api", "system", "application", "model", "service", "database",
    "server", "client", "endpoint", "function", "module", "component",
    "architecture", "interface", "framework", "library", "package",
    "pipeline", "workflow", "process", "report", "document", "prompt",
    "response", "request", "query", "result", "output", "input",
    # Adjectives / descriptors
    "fast", "efficient", "scalable", "secure", "robust", "reliable",
    "detailed", "comprehensive", "concise", "structured", "compact",
    "clear", "simple", "complex", "dynamic", "static", "custom",
    # Technical terms
    "token", "compression", "optimization", "latency", "throughput",
    "performance", "memory", "cache", "format", "schema", "json",
    "backend", "frontend", "authentication", "authorization",
]

# Pre-compute lowercase vocab for matching
_VOCAB_LOWER: List[str] = [w.lower() for w in VOCAB]

# Minimum similarity threshold for correction (0.0–1.0)
_SIMILARITY_CUTOFF: float = 0.8

# Words too short to correct (high false-positive risk)
_MIN_WORD_LENGTH: int = 3


def correct_word(word: str) -> str:
    """
    Correct a single word using fuzzy matching against VOCAB.

    Args:
        word: Input word (may be misspelled).

    Returns:
        Closest vocabulary match if similarity > 0.8, else original word.
    """
    cleaned = word.lower().strip()

    # Skip short words, numbers, and words already in vocab
    if len(cleaned) < _MIN_WORD_LENGTH:
        return word
    if cleaned in _VOCAB_LOWER:
        return word  # Already correct — preserve original casing

    matches = get_close_matches(
        cleaned, _VOCAB_LOWER, n=1, cutoff=_SIMILARITY_CUTOFF,
    )
    if matches:
        return matches[0]
    return word


def spell_check_text(text: str) -> str:
    """
    Apply spell correction to each word in the text.

    Splits on whitespace, corrects individual words, and reconstructs
    the string while preserving inter-word structure.

    Args:
        text: Input text (typically post-optimization output).

    Returns:
        Text with misspelled words corrected against VOCAB.
    """
    if not text or not text.strip():
        return text

    # Split while preserving punctuation attached to words
    # e.g. "build." → correct "build", keep "."
    tokens = text.split()
    corrected: List[str] = []

    for token in tokens:
        # Separate leading/trailing punctuation from the word core
        leading = ""
        trailing = ""
        core = token

        # Strip leading punctuation
        while core and not core[0].isalnum():
            leading += core[0]
            core = core[1:]

        # Strip trailing punctuation
        while core and not core[-1].isalnum():
            trailing = core[-1] + trailing
            core = core[:-1]

        if core:
            core = correct_word(core)

        corrected.append(f"{leading}{core}{trailing}")

    return " ".join(corrected)
