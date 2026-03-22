"""Reward utilities for deterministic RL updates."""

from __future__ import annotations

from typing import Dict, Mapping, Union


def compute_reward(
    compression_ratio: float,
    keyword_retention: float,
    semantic_score: float,
    spelling_error_rate: float,
    ambiguity: float,
    instruction_integrity: float = 1.0,
) -> Dict[str, Union[float, str]]:
    """
    Compute reward based on strict 5-dimensional piecewise metrics
    and hard penalties for constraint/meaning loss.
    """
    semantic_similarity = _clamp(semantic_score)
    
    token_reduction = 0.0
    if compression_ratio > 0.50:
        token_reduction = 1.0
    elif compression_ratio >= 0.20:
        # Interpolate 0.7 - 0.9
        token_reduction = 0.7 + ((compression_ratio - 0.20) / 0.30) * 0.2
    elif compression_ratio > 0.0:
        # Interpolate 0.3 - 0.6
        token_reduction = 0.3 + (compression_ratio / 0.20) * 0.3
        
    clarity = _clamp(1.0 - ambiguity)
    constraint_preservation = _clamp((keyword_retention + instruction_integrity) / 2.0)
    structure = _clamp(1.0 - spelling_error_rate)

    base_reward = (
        0.40 * semantic_similarity
        + 0.20 * token_reduction
        + 0.15 * clarity
        + 0.15 * constraint_preservation
        + 0.10 * structure
    )

    final_reward = base_reward
    reasoning = "Standard weighted calculation."

    if semantic_similarity < 0.8:
        final_reward = min(final_reward, 0.3)
        reasoning = "Heavy penalty: Significant loss of meaning."
    elif constraint_preservation < 1.0:
        final_reward = min(final_reward, 0.4)
        reasoning = "Heavy penalty: Constraints were lost."
    elif compression_ratio <= 0.0:
        final_reward = min(final_reward, 0.1)
        reasoning = "Heavy penalty: No token reduction."

    return {
        "semantic_similarity": round(semantic_similarity, 4),
        "token_reduction": round(token_reduction, 4),
        "clarity": round(clarity, 4),
        "constraint_preservation": round(constraint_preservation, 4),
        "structure": round(structure, 4),
        "final_reward": round(final_reward, 4),
        "reasoning": reasoning,
    }


def compute_reward_from_metrics(metrics: Mapping[str, float]) -> Dict[str, Union[float, str]]:
    """Convenience wrapper when all reward inputs are in one mapping."""
    return compute_reward(
        compression_ratio=float(metrics.get("compression_ratio", 0.0)),
        keyword_retention=float(metrics.get("keyword_retention", 0.0)),
        semantic_score=float(metrics.get("semantic_score", 0.0)),
        spelling_error_rate=float(metrics.get("spelling_error_rate", 0.0)),
        ambiguity=float(metrics.get("ambiguity", 0.0)),
        instruction_integrity=float(metrics.get("instruction_integrity", 1.0)),
    )


def estimate_spelling_error_rate(before_spellcheck: str, after_spellcheck: str) -> float:
    """Estimate spelling error rate by token differences pre/post correction."""
    before_tokens = before_spellcheck.split()
    after_tokens = after_spellcheck.split()

    if not before_tokens:
        return 0.0

    changed = 0
    for idx, token in enumerate(before_tokens):
        corrected = after_tokens[idx] if idx < len(after_tokens) else ""
        if token != corrected:
            changed += 1

    return round(_clamp(changed / len(before_tokens)), 6)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
