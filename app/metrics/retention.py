"""Keyword and instruction retention metrics."""

from typing import Set


def keyword_retention(candidate_text: str, original_keywords: Set[str]) -> float:
    """Fraction of original keywords retained in candidate text."""
    if not original_keywords:
        return 1.0

    lowered = candidate_text.lower()
    kept = sum(1 for kw in original_keywords if kw.lower() in lowered)
    return round(kept / len(original_keywords), 4)


def instruction_integrity(candidate_text: str, original_instructions: Set[str]) -> float:
    """Fraction of original instruction verbs retained in candidate text."""
    if not original_instructions:
        return 1.0

    lowered = candidate_text.lower()
    kept = sum(1 for verb in original_instructions if verb.lower() in lowered)
    return round(kept / len(original_instructions), 4)
