"""
rules.py — Stage 4: Rule Engine for Filler & Redundancy Removal.

Configurable rules dictionary for phrase replacement, adjective/adverb
removal, and redundancy elimination. Supports genome-driven toggling.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from app.utils.text_utils import (
    PHRASE_REPLACEMENTS,
    LOW_VALUE_MODIFIERS,
    normalize_whitespace,
)


# ── Conversational tokens (low-value greetings/politeness/filler) ────────────
CONVERSATIONAL_TOKENS: Set[str] = {
    "hello", "hi", "hey", "heyy",
    "thanks", "thank", "thx",
    "please", "pls", "plz",
    "bye", "goodbye",
}

# ── Important context words to preserve near conversational tokens ───────────
IMPORTANT_CONTEXT_WORDS: Set[str] = {
    "fast", "efficient", "scalable", "secure", "robust",
}


@dataclass
class RuleConfig:
    """Configuration for the rule engine, toggleable by genome."""
    remove_fillers: bool = True
    compress_phrases: bool = True
    drop_adjectives: bool = True
    preserve_keywords: bool = True
    adjective_drop_strength: float = 0.8   # 0.0–1.0
    phrase_compress_strength: float = 1.0   # 0.0–1.0


# ── Default rules (can be overridden) ────────────────────────────────────────

DEFAULT_CONFIG = RuleConfig()


def apply_rules(
    text: str,
    keywords: Set[str],
    config: RuleConfig = DEFAULT_CONFIG,
) -> str:
    """
    Apply all rule-based transformations to text.

    Args:
        text: Input text (already cleaned).
        keywords: Set of keywords to preserve.
        config: Rule configuration (toggleable).

    Returns:
        Rule-processed text.
    """
    if config.compress_phrases:
        text = _apply_phrase_replacements(text, config.phrase_compress_strength)

    if config.drop_adjectives:
        text = _remove_low_value_modifiers(
            text, keywords, config.adjective_drop_strength
        )

    text = _remove_redundant_clauses(text)
    text = normalize_whitespace(text)
    return text


def _apply_phrase_replacements(text: str, strength: float = 1.0) -> str:
    """
    Replace verbose phrases with concise equivalents.
    Strength controls how many replacements to apply (1.0 = all).
    """
    if strength <= 0.0:
        return text

    count = max(1, int(len(PHRASE_REPLACEMENTS) * strength))
    active = PHRASE_REPLACEMENTS[:count]

    for old_phrase, new_phrase in active:
        pattern = re.compile(re.escape(old_phrase), re.IGNORECASE)
        text = pattern.sub(new_phrase, text)

    return text


def _remove_low_value_modifiers(
    text: str,
    keywords: Set[str],
    strength: float = 0.8,
) -> str:
    """
    Remove low-semantic-value adjectives and adverbs.
    Preserves words that are also keywords.
    """
    if strength <= 0.0:
        return text

    # Determine which modifiers to drop based on strength
    all_mods = sorted(LOW_VALUE_MODIFIERS)
    count = max(1, int(len(all_mods) * strength))
    active_mods = set(all_mods[:count])

    words = text.split()
    result: List[str] = []

    for word in words:
        cleaned = word.lower().strip(".,!?;:\"'()-")
        # Keep if it's a keyword
        if cleaned in keywords:
            result.append(word)
            continue
        # Drop if it's a low-value modifier
        if cleaned in active_mods:
            continue
        result.append(word)

    return ' '.join(result)


def _remove_redundant_clauses(text: str) -> str:
    """Remove common redundant clause patterns."""
    # "that is to say" → ""
    text = re.sub(r'\bthat is to say\b', '', text, flags=re.IGNORECASE)

    # "what I mean is" → ""
    text = re.sub(r'\bwhat I mean is\b', '', text, flags=re.IGNORECASE)

    # "as I mentioned before/earlier" → ""
    text = re.sub(
        r'\bas I mentioned\s+(before|earlier|previously)\b',
        '', text, flags=re.IGNORECASE,
    )

    # "it goes without saying" → ""
    text = re.sub(
        r'\bit goes without saying( that)?\b',
        '', text, flags=re.IGNORECASE,
    )

    # Remove consecutive duplicate sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen: Set[str] = set()
    unique: List[str] = []
    for sent in sentences:
        normalized = sent.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(sent)

    return ' '.join(unique)
