"""
evaluator.py — Stage 11: Evaluation Engine (No ML).

Evaluates optimized prompt candidates using deterministic metrics:
- Keyword retention
- Instruction integrity
- Information density
- N-gram overlap
- Ambiguity penalty
- Redundancy score
- Concepts preserved
- Agreement score (NEW — self-consistency)

Updated fitness function (7-term):
  fitness = α·compression + β·keyword_retention + γ·instruction_integrity
          + δ·information_density + η·agreement − ε·redundancy − ζ·ambiguity
"""

from typing import Dict, List, Set, Tuple

from app.utils.text_utils import (
    generate_ngrams,
    is_meaningful_token,
    STOPWORDS,
)
from app.core.tokenizer import tokenize


def evaluate_candidate(
    original_text: str,
    candidate_text: str,
    original_keywords: Set[str],
    original_instructions: Set[str],
    agreement_score_override: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate a candidate prompt against the original.

    Args:
        agreement_score_override: Pre-computed agreement score from
            self-consistency selection (0.0–1.0). Injected externally
            because agreement is a cross-candidate metric.

    Returns:
        Dict with:
        - keyword_retention (0.0–1.0)
        - instruction_integrity (0.0–1.0)
        - information_density (0.0–1.0)
        - ngram_overlap (0.0–1.0)
        - ambiguity_penalty (0.0–1.0, lower is better)
        - compression_ratio (0.0–1.0)
        - redundancy_score (0.0–1.0, lower is better)
        - concepts_preserved (0.0–1.0, higher is better)
        - agreement_score (0.0–1.0, higher is better)
    """
    original_tokens = tokenize(original_text)
    candidate_tokens = tokenize(candidate_text)

    kr = keyword_retention(candidate_text, original_keywords)
    ii = instruction_integrity(candidate_text, original_instructions)
    density = information_density(candidate_tokens)
    ngram = ngram_overlap(original_tokens, candidate_tokens, n=2)
    ambiguity = ambiguity_penalty(candidate_text, original_instructions)
    compression = compression_ratio(original_text, candidate_text)
    redundancy = redundancy_score(candidate_text)
    concepts = concepts_preserved(original_text, candidate_text, original_keywords)

    return {
        "keyword_retention": round(kr, 4),
        "instruction_integrity": round(ii, 4),
        "information_density": round(density, 4),
        "ngram_overlap": round(ngram, 4),
        "ambiguity_penalty": round(ambiguity, 4),
        "compression_ratio": round(compression, 4),
        "redundancy_score": round(redundancy, 4),
        "concepts_preserved": round(concepts, 4),
        "agreement_score": round(agreement_score_override, 4),
    }


def fitness_score(
    metrics: Dict[str, float],
    alpha: float = 0.25,
    beta: float = 0.20,
    gamma: float = 0.20,
    delta: float = 0.10,
    epsilon: float = 0.10,
    zeta: float = 0.05,
    eta: float = 0.10,
) -> float:
    """
    Compute weighted fitness score from evaluation metrics.

    Updated 7-term formula:
    fitness = α·compression + β·keyword_retention + γ·instruction_integrity
            + δ·information_density + η·agreement
            − ε·redundancy_score − ζ·ambiguity_penalty
    """
    score = (
        alpha * metrics["compression_ratio"]
        + beta * metrics["keyword_retention"]
        + gamma * metrics["instruction_integrity"]
        + delta * metrics["information_density"]
        + eta * metrics.get("agreement_score", 0.0)
        - epsilon * metrics.get("redundancy_score", 0.0)
        - zeta * metrics["ambiguity_penalty"]
    )
    return round(max(0.0, min(1.0, score)), 4)


# ── Individual metrics ───────────────────────────────────────────────────────

def keyword_retention(candidate_text: str, original_keywords: Set[str]) -> float:
    """
    Measure what fraction of original keywords are retained in candidate.
    """
    if not original_keywords:
        return 1.0

    candidate_lower = candidate_text.lower()
    kept = sum(1 for kw in original_keywords if kw.lower() in candidate_lower)
    return kept / len(original_keywords)


def instruction_integrity(
    candidate_text: str,
    original_instructions: Set[str],
) -> float:
    """
    Ensure core instruction verbs are still present in the candidate.
    """
    if not original_instructions:
        return 1.0

    candidate_lower = candidate_text.lower()
    kept = sum(
        1 for verb in original_instructions
        if verb.lower() in candidate_lower
    )
    return kept / len(original_instructions)


def information_density(tokens: List[str]) -> float:
    """
    Compute information density: meaningful_tokens / total_tokens.
    Higher density = more content words, fewer filler words.
    """
    if not tokens:
        return 0.0

    meaningful = sum(1 for t in tokens if is_meaningful_token(t))
    return meaningful / len(tokens)


def ngram_overlap(
    original_tokens: List[str],
    candidate_tokens: List[str],
    n: int = 2,
) -> float:
    """
    Measure structural similarity using n-gram overlap (Jaccard).
    """
    orig_lower = [t.lower() for t in original_tokens]
    cand_lower = [t.lower() for t in candidate_tokens]

    orig_ngrams = set(generate_ngrams(orig_lower, n))
    cand_ngrams = set(generate_ngrams(cand_lower, n))

    if not orig_ngrams:
        return 1.0 if not cand_ngrams else 0.0

    intersection = orig_ngrams & cand_ngrams
    union = orig_ngrams | cand_ngrams

    return len(intersection) / len(union) if union else 0.0


def compression_ratio(original_text: str, candidate_text: str) -> float:
    """
    Compute compression ratio: 1 - (len(candidate) / len(original)).
    Values closer to 1.0 mean more compression.
    """
    original_len = len(original_text.split())
    candidate_len = len(candidate_text.split())

    if original_len == 0:
        return 0.0

    return max(0.0, 1.0 - (candidate_len / original_len))


def ambiguity_penalty(
    candidate_text: str,
    original_instructions: Set[str],
) -> float:
    """
    Heuristic ambiguity penalty:
    - Penalize if candidate is too short (< 5 words)
    - Penalize if key instruction verbs are missing
    - Penalize if sentence is incomplete (no period/verb)
    """
    penalty = 0.0
    words = candidate_text.split()

    # Too short → likely lost meaning
    if len(words) < 5:
        penalty += 0.3

    # Missing instructions
    if original_instructions:
        candidate_lower = candidate_text.lower()
        missing = sum(
            1 for v in original_instructions
            if v.lower() not in candidate_lower
        )
        penalty += 0.3 * (missing / len(original_instructions))

    # No verb-like word at all (very rough check)
    has_verb = any(
        w.lower().endswith(('ate', 'ize', 'ify', 'ise', 'ing', 'ed'))
        or w.lower() in {'do', 'run', 'get', 'set', 'use', 'add', 'put',
                         'make', 'give', 'take', 'show', 'list', 'find'}
        for w in words
    )
    if not has_verb and len(words) > 3:
        penalty += 0.2

    return min(1.0, penalty)


# ── New semantic-level metrics ───────────────────────────────────────────────

def redundancy_score(text: str) -> float:
    """
    Estimate residual redundancy in the candidate text.

    Measures repeated meaningful words (excluding stopwords).
    Score = (total_meaningful - unique_meaningful) / total_meaningful.
    0.0 = no redundancy, 1.0 = fully redundant.
    """
    words = [
        w.lower().strip(".,!?;:\"'()-")
        for w in text.split()
    ]
    meaningful = [w for w in words if w and w not in STOPWORDS and len(w) > 2]

    if not meaningful:
        return 0.0

    unique = set(meaningful)
    duplicates = len(meaningful) - len(unique)
    return duplicates / len(meaningful)


def concepts_preserved(
    original_text: str,
    candidate_text: str,
    keywords: Set[str],
) -> float:
    """
    Measure what fraction of semantic concepts from the original
    are retained in the candidate.

    Concepts = unique meaningful words (non-stopwords, >2 chars).
    """
    def _extract_concepts(text: str) -> Set[str]:
        words = text.lower().split()
        return {
            w.strip(".,!?;:\"'()-")
            for w in words
            if len(w.strip(".,!?;:\"'()-")) > 2
            and w.strip(".,!?;:\"'()-") not in STOPWORDS
        }

    original_concepts = _extract_concepts(original_text)
    candidate_concepts = _extract_concepts(candidate_text)

    if not original_concepts:
        return 1.0

    retained = original_concepts & candidate_concepts
    # Boost score for keyword retention
    keyword_bonus = sum(
        0.5 for kw in keywords
        if kw.lower() in candidate_concepts and kw.lower() in original_concepts
    )

    raw_score = (len(retained) + keyword_bonus) / (len(original_concepts) + keyword_bonus)
    return min(1.0, raw_score)
