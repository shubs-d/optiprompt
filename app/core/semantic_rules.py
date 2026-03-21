"""
semantic_rules.py — Semantic Deduplication Engine.

Extends existing synonym-cluster normalization (semantic.py) with
cross-sentence deduplication, redundant modifier chain removal,
and sentence-level similarity-based merging.

All logic is deterministic — no ML, no embeddings.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from app.core.semantic import (
    SYNONYM_CLUSTERS,
    _WORD_TO_CLUSTER,
    _CLUSTER_REPS,
    normalize_concepts,
)
from app.utils.text_utils import (
    LOW_VALUE_MODIFIERS,
    STOPWORDS,
    normalize_whitespace,
    split_sentences,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeduplicationMetrics:
    """Metrics from semantic deduplication pass."""
    duplicate_concepts_removed: int = 0      # cross-sentence duplicate concepts removed
    redundant_modifier_chains: int = 0       # stacked modifier chains collapsed
    similar_sentences_merged: int = 0        # near-duplicate sentences eliminated
    total_redundancy_score: float = 0.0      # overall redundancy fraction

    def to_dict(self) -> Dict[str, float]:
        return {
            "duplicate_concepts_removed": self.duplicate_concepts_removed,
            "redundant_modifier_chains": self.redundant_modifier_chains,
            "similar_sentences_merged": self.similar_sentences_merged,
            "total_redundancy_score": round(self.total_redundancy_score, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CROSS-SENTENCE CONCEPT DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_content_words(text: str) -> Set[str]:
    """Extract meaningful content words (non-stopword, >2 chars)."""
    return {
        w.lower().strip(".,!?;:\"'()-")
        for w in text.split()
        if len(w.strip(".,!?;:\"'()-")) > 2
        and w.lower().strip(".,!?;:\"'()-") not in STOPWORDS
    }


def deduplicate_concepts_across_sentences(
    text: str,
    keywords: Set[str],
    threshold: float = 0.5,
) -> Tuple[str, int]:
    """
    Remove repeated non-keyword content words across sentences.

    If a content word appears in multiple sentences and is NOT
    a keyword, keep only its first occurrence (remove subsequent).

    Args:
        text: Input text.
        keywords: Keywords to always preserve.
        threshold: 0.0–1.0 controlling aggressiveness.
                   Higher = more aggressive deduplication.

    Returns:
        (deduplicated_text, num_concepts_removed)
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text, 0

    # Track which content words we've already seen
    seen_concepts: Set[str] = set()
    result_sentences: List[str] = []
    removed = 0

    # Build keyword lower set for fast lookup
    kw_lower = {k.lower() for k in keywords}

    for sent in sentences:
        words = sent.split()
        kept: List[str] = []

        for word in words:
            cleaned = word.lower().strip(".,!?;:\"'()-")

            # Always keep keywords, short words, and stopwords
            if (not cleaned
                    or len(cleaned) <= 2
                    or cleaned in STOPWORDS
                    or cleaned in kw_lower):
                kept.append(word)
                continue

            # Normalize via synonym cluster representative
            if cleaned in _WORD_TO_CLUSTER:
                _, rep = _WORD_TO_CLUSTER[cleaned]
                concept_key = rep
            else:
                concept_key = cleaned

            # If we've seen this concept before and threshold allows removal
            if concept_key in seen_concepts and threshold >= 0.3:
                removed += 1
                continue

            seen_concepts.add(concept_key)
            kept.append(word)

        result_sentences.append(" ".join(kept))

    return " ".join(result_sentences), removed


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REDUNDANT MODIFIER CHAIN REMOVAL
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns: sequences of 2+ low-value modifiers stacked before a content word
_MODIFIER_CHAIN_RE = re.compile(
    r'\b('
    + '|'.join(re.escape(m) for m in sorted(LOW_VALUE_MODIFIERS, key=len, reverse=True))
    + r')\s+('
    + '|'.join(re.escape(m) for m in sorted(LOW_VALUE_MODIFIERS, key=len, reverse=True))
    + r')\b',
    re.IGNORECASE,
)


def collapse_modifier_chains(text: str) -> Tuple[str, int]:
    """
    Collapse stacked modifier chains.

    "very extremely important" → "important"
    "really quite completely" → (removed, keeps the next non-modifier word)

    Returns:
        (collapsed_text, num_chains_collapsed)
    """
    chains_collapsed = 0

    # Iteratively collapse until stable (max 5 passes)
    for _ in range(5):
        new_text = _MODIFIER_CHAIN_RE.sub(
            lambda m: m.group(2),  # keep second modifier (will collapse again if chained)
            text,
        )
        if new_text == text:
            break
        chains_collapsed += 1
        text = new_text

    return normalize_whitespace(text), chains_collapsed


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SENTENCE-LEVEL SIMILARITY DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def _sentence_jaccard(sent_a: str, sent_b: str) -> float:
    """Compute Jaccard similarity between content words of two sentences."""
    words_a = _extract_content_words(sent_a)
    words_b = _extract_content_words(sent_b)

    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def deduplicate_similar_sentences(
    text: str,
    similarity_threshold: float = 0.7,
) -> Tuple[str, int]:
    """
    Remove near-duplicate sentences using Jaccard similarity on content words.

    When two sentences have Jaccard similarity ≥ threshold, keep only the
    longer one (more informative).

    Args:
        text: Input text.
        similarity_threshold: 0.0–1.0. Higher = only exact duplicates removed.

    Returns:
        (deduplicated_text, sentences_removed)
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text, 0

    keep_mask = [True] * len(sentences)
    removed = 0

    for i in range(len(sentences)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(sentences)):
            if not keep_mask[j]:
                continue

            sim = _sentence_jaccard(sentences[i], sentences[j])
            if sim >= similarity_threshold:
                # Keep the longer sentence (more informative)
                if len(sentences[j].split()) <= len(sentences[i].split()):
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                removed += 1
                break  # move to next i

    result = [s for s, keep in zip(sentences, keep_mask) if keep]
    return " ".join(result), removed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def semantic_deduplicate(
    text: str,
    keywords: Set[str],
    aggressiveness: float = 0.5,
) -> Tuple[str, DeduplicationMetrics]:
    """
    Full semantic deduplication pipeline.

    Applies three deduplication strategies in sequence:
    1. Cross-sentence concept deduplication
    2. Redundant modifier chain collapse
    3. Sentence-level similarity deduplication

    Args:
        text: Pre-processed text.
        keywords: Keywords to preserve.
        aggressiveness: 0.0–1.0 controlling overall aggressiveness.

    Returns:
        (deduplicated_text, metrics)
    """
    metrics = DeduplicationMetrics()
    original_word_count = len(text.split())

    # Step 1: Cross-sentence concept dedup
    text, concept_removed = deduplicate_concepts_across_sentences(
        text, keywords, threshold=aggressiveness,
    )
    metrics.duplicate_concepts_removed = concept_removed

    # Step 2: Modifier chain collapse
    text, chains = collapse_modifier_chains(text)
    metrics.redundant_modifier_chains = chains

    # Step 3: Sentence-level dedup (only at higher aggressiveness)
    if aggressiveness >= 0.4:
        sim_threshold = 0.8 - (aggressiveness * 0.2)  # range: 0.6–0.8
        sim_threshold = max(0.55, min(0.85, sim_threshold))
        text, sent_removed = deduplicate_similar_sentences(
            text, similarity_threshold=sim_threshold,
        )
        metrics.similar_sentences_merged = sent_removed

    # Overall redundancy score
    new_word_count = len(text.split())
    total_removed = original_word_count - new_word_count
    metrics.total_redundancy_score = (
        total_removed / max(1, original_word_count)
    )

    text = normalize_whitespace(text)
    return text, metrics
