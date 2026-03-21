"""
transformations.py — Stage 8: Multi-Candidate Generation + Self-Consistency.

Generates 3+ candidate variants (aggressive, balanced, structure-preserving)
by applying the pipeline stages with different rule intensities.

Includes self-consistency selection:
- Compute pairwise agreement scores (Jaccard on tokens)
- Select candidate with best combined fitness + agreement
"""

from typing import Dict, List, Set, Tuple

from app.core.cleaner import clean
from app.core.compressor import compress
from app.core.rules import RuleConfig, apply_rules
from app.core.genome import Genome, AGGRESSIVE_GENOME, BALANCED_GENOME, SAFE_GENOME
from app.core.semantic import SemanticConfig, semantic_compress
from app.utils.text_utils import normalize_whitespace


# ── Variant definitions ──────────────────────────────────────────────────────

VARIANT_CONFIGS: Dict[str, Tuple[Genome, str]] = {
    "aggressive": (AGGRESSIVE_GENOME, "Maximum compression, minimal structure"),
    "balanced": (BALANCED_GENOME, "Balance between compression and clarity"),
    "structure_preserving": (SAFE_GENOME, "Minimal changes, maximum clarity"),
}


def generate_candidates(
    text: str,
    keywords: Set[str],
    instructions: Set[str],
) -> List[Dict]:
    """
    Generate multiple candidate optimizations with different intensities.

    Args:
        text: Cleaned, tokenized text (post-stages 1-3).
        keywords: Extracted keyword set.
        instructions: Extracted instruction set.

    Returns:
        List of dicts, each with:
        - 'name': variant name
        - 'text': optimized text
        - 'genome': genome used
        - 'description': brief description
    """
    candidates: List[Dict] = []

    for name, (genome, description) in VARIANT_CONFIGS.items():
        optimized = apply_genome(text, keywords, genome)
        candidates.append({
            "name": name,
            "text": optimized,
            "genome": genome,
            "description": description,
        })

    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-CONSISTENCY SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize_to_set(text: str) -> Set[str]:
    """Tokenize text into a lowercase word set (for agreement computation)."""
    return {w.lower().strip(".,!?;:\"'()-") for w in text.split() if w.strip()}


def compute_agreement_scores(candidates: List[Dict]) -> List[float]:
    """
    Compute per-candidate agreement scores via pairwise Jaccard overlap.

    For each candidate, the agreement score is the average Jaccard
    similarity with all other candidates:

        agreement(i) = mean( |tokens_i ∩ tokens_j| / |tokens_i ∪ tokens_j| )
        for all j ≠ i

    A high agreement score means the candidate is consistent with the
    consensus of all variants — a sign of stable semantic content.

    Args:
        candidates: List of candidate dicts with 'text' field.

    Returns:
        List of agreement scores (same order as input candidates).
    """
    if len(candidates) <= 1:
        return [1.0] * len(candidates)

    # Pre-compute token sets
    token_sets = [_tokenize_to_set(c["text"]) for c in candidates]

    scores: List[float] = []
    for i, tokens_i in enumerate(token_sets):
        pairwise: List[float] = []
        for j, tokens_j in enumerate(token_sets):
            if i == j:
                continue
            union = tokens_i | tokens_j
            if not union:
                pairwise.append(1.0)
            else:
                intersection = tokens_i & tokens_j
                pairwise.append(len(intersection) / len(union))
        # Average pairwise Jaccard
        avg = sum(pairwise) / len(pairwise) if pairwise else 1.0
        scores.append(round(avg, 4))

    return scores


def select_by_consensus(
    candidates: List[Dict],
    fitness_weight: float = 0.6,
    agreement_weight: float = 0.4,
) -> Tuple[Dict, float]:
    """
    Select the best candidate using combined fitness + agreement scoring.

    consensus_score = fitness_weight * fitness + agreement_weight * agreement

    Args:
        candidates: List of candidate dicts with 'text' and 'fitness' fields.
        fitness_weight: Weight for fitness in combined score.
        agreement_weight: Weight for agreement in combined score.

    Returns:
        (best_candidate, best_agreement_score)
    """
    agreement_scores = compute_agreement_scores(candidates)

    best_idx = 0
    best_consensus = -1.0
    best_agreement = 0.0

    for i, (cand, agreement) in enumerate(zip(candidates, agreement_scores)):
        fitness = cand.get("fitness", 0.0)
        consensus = fitness_weight * fitness + agreement_weight * agreement

        if consensus > best_consensus:
            best_consensus = consensus
            best_idx = i
            best_agreement = agreement

    return candidates[best_idx], best_agreement


# ═══════════════════════════════════════════════════════════════════════════════
# GENOME APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_genome(
    text: str,
    keywords: Set[str],
    genome: Genome,
) -> str:
    """
    Apply a genome's parameters to transform text through the pipeline.

    This maps genome genes to rule config, semantic config, and
    compression parameters.
    """
    # Blend structure preference into effective compression intensity.
    # Higher structure_weight keeps more of the original form.
    effective_compression = max(
        0.0,
        min(1.0, genome.compression_level * (1.0 - (0.5 * genome.structure_weight))),
    )

    # Build rule config from genome
    config = RuleConfig(
        remove_fillers=genome.filler_removal_strength > 0.1,
        compress_phrases=effective_compression > 0.1,
        drop_adjectives=genome.filler_removal_strength > 0.3,
        preserve_keywords=genome.keyword_preservation_bias > 0.3,
        adjective_drop_strength=genome.filler_removal_strength,
        phrase_compress_strength=effective_compression,
    )

    # Stage 4: Apply rules
    result = apply_rules(text, keywords, config)

    # Stage 5: Apply semantic compression (driven by genome genes)
    sem_config = SemanticConfig(
        redundancy_threshold=genome.redundancy_threshold,
        modifier_pruning_level=genome.modifier_pruning_level,
        enable_synonym_collapse=genome.redundancy_threshold > 0.15,
        enable_phrase_compression=effective_compression > 0.15,
        enable_concept_pruning=genome.modifier_pruning_level > 0.2,
    )
    result, _ = semantic_compress(result, keywords, config=sem_config)

    # Stage 6: Apply clause compression
    result = compress(result, level=effective_compression)

    if config.preserve_keywords:
        result = _restore_missing_keywords(result, keywords, genome.keyword_preservation_bias)

    # Final cleanup
    result = normalize_whitespace(result)

    return result


def _restore_missing_keywords(text: str, keywords: Set[str], bias: float) -> str:
    """
    Re-attach missing high-value keywords when preservation bias is strong.
    Deterministic ordering keeps output stable.
    """
    if not keywords or bias < 0.5:
        return text

    lower_text = text.lower()
    missing = [kw for kw in sorted(keywords) if kw.lower() not in lower_text]
    if not missing:
        return text

    # Bias controls how many missing keywords are restored.
    restore_count = max(1, int(len(missing) * min(1.0, bias)))
    to_restore = missing[:restore_count]
    return f"{text} | keep: {', '.join(to_restore)}"
