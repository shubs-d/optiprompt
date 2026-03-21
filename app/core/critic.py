"""
critic.py — Textual Critic (Feedback Loop).

Simulates TextGrad-style refinement WITHOUT ML. Detects and corrects
three classes of optimisation defects in a deterministic feedback loop:

1. Over-compression: too much content removed, keyword retention dropped
2. Lost intent: core instruction verbs missing from the output
3. Fragmentation: sentences broken into incoherent fragments

Runs up to MAX_PASSES correction passes. Same input → same output.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from app.core.tokenizer import tokenize
from app.utils.text_utils import (
    STOPWORDS,
    normalize_whitespace,
    split_sentences,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_PASSES = 3                # maximum correction iterations

# Thresholds for defect detection
OVER_COMPRESSION_RATIO = 0.55     # compression > this triggers relaxation
MIN_KEYWORD_RETENTION = 0.65      # keyword retention below this triggers keyword reinsertion
MIN_INSTRUCTION_INTEGRITY = 0.60  # instruction integrity below this triggers verb reinsertion
MIN_AVG_SENTENCE_LENGTH = 3       # avg words/sentence below this triggers fragment merging


# ═══════════════════════════════════════════════════════════════════════════════
# DEFECT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CriticReport:
    """Diagnostic report from the textual critic."""
    passes_applied: int = 0
    over_compression_detected: bool = False
    lost_intent_detected: bool = False
    fragmentation_detected: bool = False
    keywords_reinserted: List[str] = field(default_factory=list)
    verbs_reinserted: List[str] = field(default_factory=list)
    fragments_merged: int = 0

    def to_dict(self) -> Dict:
        return {
            "passes_applied": self.passes_applied,
            "over_compression_detected": self.over_compression_detected,
            "lost_intent_detected": self.lost_intent_detected,
            "fragmentation_detected": self.fragmentation_detected,
            "keywords_reinserted": self.keywords_reinserted,
            "verbs_reinserted": self.verbs_reinserted,
            "fragments_merged": self.fragments_merged,
        }


def _compute_keyword_retention(text: str, keywords: Set[str]) -> float:
    """Fraction of keywords present in text."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    kept = sum(1 for kw in keywords if kw.lower() in text_lower)
    return kept / len(keywords)


def _compute_instruction_integrity(text: str, instructions: Set[str]) -> float:
    """Fraction of instruction verbs present in text."""
    if not instructions:
        return 1.0
    text_lower = text.lower()
    kept = sum(1 for v in instructions if v.lower() in text_lower)
    return kept / len(instructions)


def _compute_compression_ratio(original: str, candidate: str) -> float:
    """1.0 - (candidate_words / original_words). Higher = more compressed."""
    orig_len = len(original.split())
    cand_len = len(candidate.split())
    if orig_len == 0:
        return 0.0
    return max(0.0, 1.0 - (cand_len / orig_len))


def _avg_sentence_length(text: str) -> float:
    """Average number of words per sentence."""
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    total_words = sum(len(s.split()) for s in sentences)
    return total_words / len(sentences)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECTION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

def _relax_over_compression(
    text: str,
    keywords: Set[str],
    report: CriticReport,
) -> str:
    """
    Re-insert missing keywords to counter over-compression.

    Strategy: append missing high-value keywords at the end,
    sorted deterministically for stable output.
    """
    text_lower = text.lower()
    missing = sorted(kw for kw in keywords if kw.lower() not in text_lower)

    if not missing:
        return text

    # Re-insert top missing keywords (up to 5)
    to_reinsert = missing[:5]
    report.keywords_reinserted.extend(to_reinsert)

    # Append as a concise clause
    suffix = ", ".join(to_reinsert)
    return f"{text.rstrip('.')} — {suffix}."


def _restore_lost_intent(
    text: str,
    instructions: Set[str],
    report: CriticReport,
) -> str:
    """
    Re-insert missing instruction verbs.

    Strategy: prepend missing action verbs at the start of the text
    to restore imperative intent.
    """
    text_lower = text.lower()
    missing = sorted(v for v in instructions if v.lower() not in text_lower)

    if not missing:
        return text

    # Re-insert top missing verbs (up to 3)
    to_reinsert = missing[:3]
    report.verbs_reinserted.extend(to_reinsert)

    # Prepend as action directives
    prefix = ", ".join(v.capitalize() for v in to_reinsert)
    # If text already starts with a capital, lowercase it for flow
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return f"{prefix}: {text}"


def _merge_fragments(
    text: str,
    report: CriticReport,
) -> str:
    """
    Merge short sentence fragments into coherent sentences.

    Strategy: join consecutive fragments (<= MIN_AVG_SENTENCE_LENGTH words)
    separated by commas instead of periods.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text

    merged: List[str] = []
    buffer: List[str] = []
    merges = 0

    for sent in sentences:
        word_count = len(sent.split())
        if word_count <= MIN_AVG_SENTENCE_LENGTH:
            # Short fragment → accumulate in buffer
            # Strip trailing period for merging
            buffer.append(sent.rstrip(".!?").strip())
        else:
            # Flush buffer if any
            if buffer:
                # Merge buffer fragments with current sentence
                combined = ", ".join(buffer)
                merged.append(f"{combined}, {sent.rstrip('.').strip()}.")
                merges += len(buffer)
                buffer = []
            else:
                merged.append(sent)

    # Flush remaining buffer
    if buffer:
        merged.append(", ".join(buffer) + ".")
        if len(buffer) > 1:
            merges += len(buffer) - 1

    report.fragments_merged = merges
    return " ".join(merged)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def refine(
    candidate_text: str,
    original_text: str,
    keywords: Set[str],
    instructions: Set[str],
) -> Tuple[str, CriticReport]:
    """
    Run the textual critic feedback loop.

    Detects defects (over-compression, lost intent, fragmentation)
    and applies deterministic corrections for up to MAX_PASSES iterations.

    Args:
        candidate_text: The optimised prompt candidate.
        original_text: The original input prompt (for ratio computation).
        keywords: Extracted keyword set.
        instructions: Extracted instruction verb set.

    Returns:
        (refined_text, critic_report)
    """
    report = CriticReport()
    text = candidate_text

    for pass_num in range(MAX_PASSES):
        changed = False

        # ── Check 1: Over-compression ────────────────────────────────────
        compression = _compute_compression_ratio(original_text, text)
        kr = _compute_keyword_retention(text, keywords)

        if compression > OVER_COMPRESSION_RATIO and kr < MIN_KEYWORD_RETENTION:
            report.over_compression_detected = True
            text = _relax_over_compression(text, keywords, report)
            changed = True

        # ── Check 2: Lost intent ─────────────────────────────────────────
        ii = _compute_instruction_integrity(text, instructions)

        if ii < MIN_INSTRUCTION_INTEGRITY:
            report.lost_intent_detected = True
            text = _restore_lost_intent(text, instructions, report)
            changed = True

        # ── Check 3: Fragmentation ───────────────────────────────────────
        avg_len = _avg_sentence_length(text)

        if avg_len < MIN_AVG_SENTENCE_LENGTH and len(text.split()) > 3:
            report.fragmentation_detected = True
            text = _merge_fragments(text, report)
            changed = True

        report.passes_applied = pass_num + 1

        # If no defects detected, stop early
        if not changed:
            break

    text = normalize_whitespace(text)
    return text, report
