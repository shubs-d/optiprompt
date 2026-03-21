"""
cleaner.py — Stage 1: Regex Cleaning.

Removes extra whitespace, repeated punctuation, filler phrases,
and normalizes casing where appropriate.
"""

import re
from typing import List, Optional

from app.utils.text_utils import FILLER_PHRASES, normalize_whitespace


def clean(text: str, filler_strength: float = 1.0) -> str:
    """
    Run the full cleaning pipeline.

    Args:
        text: Raw input prompt.
        filler_strength: 0.0–1.0 controlling how aggressively fillers
                         are removed. 1.0 = remove all, 0.0 = remove none.

    Returns:
        Cleaned text.
    """
    text = _normalize_unicode(text)
    text = _collapse_whitespace(text)
    text = _fix_repeated_punctuation(text)
    text = _remove_filler_phrases(text, strength=filler_strength)
    text = _normalize_casing(text)
    text = normalize_whitespace(text)
    return text


# ── Internal helpers ─────────────────────────────────────────────────────────

def _normalize_unicode(text: str) -> str:
    """Replace fancy quotes, dashes, ellipses with ASCII equivalents."""
    replacements = {
        '\u2018': "'", '\u2019': "'",   # smart single quotes
        '\u201c': '"', '\u201d': '"',   # smart double quotes
        '\u2013': '-', '\u2014': '-',   # en/em dashes
        '\u2026': '...',                 # ellipsis
        '\u00a0': ' ',                   # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces, strip edges."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _fix_repeated_punctuation(text: str) -> str:
    """Reduce repeated punctuation (e.g., '!!!' → '!', '...' → '.')."""
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r';{2,}', ';', text)
    text = re.sub(r':{2,}', ':', text)
    return text


def _remove_filler_phrases(text: str, strength: float = 1.0) -> str:
    """
    Remove filler phrases from text.

    strength controls how many fillers are removed:
    - 1.0: remove all known fillers
    - 0.5: remove ≈ top half (longest/most verbose)
    - 0.0: remove none
    """
    if strength <= 0.0:
        return text

    # Determine how many fillers to apply
    count = max(1, int(len(FILLER_PHRASES) * strength))
    active_fillers = FILLER_PHRASES[:count]  # already sorted longest-first

    text_lower = text.lower()
    for filler in active_fillers:
        # Case-insensitive removal preserving surrounding structure
        pattern = re.compile(re.escape(filler), re.IGNORECASE)
        text = pattern.sub('', text)

    # Clean up artifacts: double spaces, leading commas
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s*,\s*', '', text)
    text = re.sub(r'\s*,\s*,', ',', text)
    return text.strip()


def _normalize_casing(text: str) -> str:
    """
    Normalize shouted text (ALL CAPS segments > 3 words) to title case.
    Preserves intentional single-word caps like 'API', 'SQL', 'JSON'.
    """
    def _fix_caps(match: re.Match) -> str:
        segment = match.group(0)
        words = segment.split()
        # Only normalize if > 3 consecutive all-caps words
        if len(words) > 3:
            return segment.capitalize()
        return segment

    # Match sequences of all-caps words
    text = re.sub(
        r'\b(?:[A-Z]{2,}\s+){3,}[A-Z]{2,}\b',
        _fix_caps,
        text,
    )
    return text


# ── Pre-optimization normalization functions ─────────────────────────────────

# Slang → standard word mapping
_SLANG_MAP = {
    "plz": "please",
    "pls": "please",
    "thx": "thanks",
    "thnx": "thanks",
    "u": "you",
    "ur": "your",
    "r": "are",
    "b4": "before",
    "bc": "because",
    "w/": "with",
    "w/o": "without",
    "idk": "i don't know",
    "msg": "message",
    "info": "information",
}


def normalize_elongation(text: str) -> str:
    """
    Reduce repeated characters to at most 2 occurrences.

    Examples:
        "heyyyy"  → "heyy"
        "goooood" → "good"
        "hellllo" → "hello"
    """
    # Collapse 3+ repeated characters down to 2
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def normalize_slang(text: str) -> str:
    """
    Map common slang/abbreviations to their standard forms.

    Also handles residual elongation (e.g. "plzz" → "plz" → "please")
    by progressively de-duplicating trailing repeated chars.
    """
    words = text.split()
    result = []
    for word in words:
        # Preserve punctuation around the word
        lower = word.lower().strip(".,!?;:\"'()-")
        if lower in _SLANG_MAP:
            result.append(_SLANG_MAP[lower])
        else:
            # Try de-elongating: remove trailing repeated chars progressively
            # e.g. "plzz" → "plz", "thxx" → "thx"
            de_elongated = re.sub(r'(.)\1+$', r'\1', lower)
            if de_elongated != lower and de_elongated in _SLANG_MAP:
                result.append(_SLANG_MAP[de_elongated])
            else:
                result.append(word)
    return " ".join(result)


def remove_conversational_tokens(text: str) -> str:
    """
    Remove low-value conversational tokens (greetings, politeness, etc.)
    while preserving IMPORTANT_CONTEXT_WORDS when they appear before
    noun-like content.

    Uses CONVERSATIONAL_TOKENS and IMPORTANT_CONTEXT_WORDS from rules.py.
    """
    from app.core.rules import CONVERSATIONAL_TOKENS, IMPORTANT_CONTEXT_WORDS

    words = text.split()
    result = []

    for i, word in enumerate(words):
        cleaned = word.lower().strip(".,!?;:\"'()-")

        # Always keep important context words
        if cleaned in IMPORTANT_CONTEXT_WORDS:
            result.append(word)
            continue

        # Remove conversational tokens
        if cleaned in CONVERSATIONAL_TOKENS:
            continue

        result.append(word)

    return " ".join(result)


def _remove_repeated_words(text: str) -> str:
    """
    Remove consecutive duplicate words.

    Example: "build build a system" → "build a system"
    """
    words = text.split()
    if not words:
        return text

    deduped = [words[0]]
    for word in words[1:]:
        if word.lower() != deduped[-1].lower():
            deduped.append(word)

    return " ".join(deduped)


def normalize_text(text: str) -> str:
    """
    Unified pre-optimization text normalization.

    Pipeline:
        1. Lowercase
        2. Normalize elongated characters (e.g. "heyyyy" → "heyy")
        3. Normalize slang (e.g. "plz" → "please")
        4. Remove conversational tokens (greetings, politeness)
        5. Remove consecutive duplicate words

    Args:
        text: Raw input prompt.

    Returns:
        Normalized text ready for the optimization pipeline.
    """
    text = text.lower()
    text = normalize_elongation(text)
    text = normalize_slang(text)
    text = remove_conversational_tokens(text)
    text = _remove_repeated_words(text)
    # Clean up any extra whitespace introduced
    text = re.sub(r'\s+', ' ', text).strip()
    return text

