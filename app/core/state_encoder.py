"""State encoder for deterministic RL strategy selection."""

from __future__ import annotations

import re
from typing import Dict


_INTENT_VERB_BUCKETS = {
    "creation": {
        "create", "build", "generate", "write", "design", "develop", "implement",
    },
    "analysis": {
        "analyze", "explain", "summarize", "review", "compare", "evaluate", "classify",
    },
    "optimization": {
        "optimize", "improve", "reduce", "compress", "refactor", "debug", "fix",
    },
}


def encode_state(prompt_text: str, intent_dict: Dict[str, list]) -> str:
    """Encode prompt context to a compact, discrete state key.

    State dimensions:
    - prompt length bucket
    - inferred intent type
    - noise level bucket
    """
    length_bucket = _bucket_prompt_length(prompt_text)
    intent_type = _infer_intent_type(intent_dict)
    noise_bucket = _bucket_noise_level(prompt_text)
    return f"len:{length_bucket}|intent:{intent_type}|noise:{noise_bucket}"


def _bucket_prompt_length(text: str) -> str:
    word_count = len([w for w in text.split() if w.strip()])
    if word_count <= 12:
        return "short"
    if word_count <= 35:
        return "medium"
    return "long"


def _infer_intent_type(intent_dict: Dict[str, list]) -> str:
    actions = [str(a).lower() for a in intent_dict.get("actions", [])]
    if not actions:
        return "general"

    for intent_name, verbs in _INTENT_VERB_BUCKETS.items():
        if any(action in verbs for action in actions):
            return intent_name

    return "general"


def _bucket_noise_level(text: str) -> str:
    if not text:
        return "low"

    punct_count = len(re.findall(r"[^\w\s]", text))
    upper_count = sum(1 for ch in text if ch.isalpha() and ch.isupper())
    char_count = max(1, len(text))

    punct_ratio = punct_count / char_count
    upper_ratio = upper_count / char_count

    elongated = len(re.findall(r"(.)\1{2,}", text.lower()))
    elongation_bonus = min(0.25, elongated * 0.05)

    noise_score = (0.6 * punct_ratio) + (0.4 * upper_ratio) + elongation_bonus

    if noise_score < 0.05:
        return "low"
    if noise_score < 0.12:
        return "medium"
    return "high"
