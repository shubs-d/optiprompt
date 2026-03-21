"""
intent_graph.py — Standalone Intent Graph Extraction Module.

Provides high-level intent extraction and prompt reconstruction
from structured semantic representations.

Wraps and extends the low-level extraction in semantic.py:
- extract_full_intent(): graph + prepositional constraints
- reconstruct_from_graph(): rebuild a compressed prompt from the graph
- Intent dict: {actions, objects, constraints, modifiers}

No ML, no embeddings — fully deterministic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from app.core.semantic import (
    IntentGraph,
    extract_constraints,
    extract_intent_graph,
)
from app.utils.text_utils import normalize_whitespace


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FullIntent:
    """Complete intent representation including prepositional constraints."""
    graph: IntentGraph
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        """Export to the canonical intent dict format."""
        return {
            "actions": list(dict.fromkeys(self.graph.actions)),       # deduplicated, ordered
            "objects": list(dict.fromkeys(self.graph.objects)),
            "constraints": list(dict.fromkeys(self.constraints)),
            "modifiers": list(dict.fromkeys(self.graph.modifiers)),
        }

    @property
    def unique_action_count(self) -> int:
        return len(set(self.graph.actions))

    @property
    def unique_object_count(self) -> int:
        return len(set(self.graph.objects))


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_full_intent(
    text: str,
    keywords: Set[str],
) -> FullIntent:
    """
    Extract a complete intent representation from text.

    Combines token-level intent graph extraction with
    prepositional-phrase constraint extraction.

    Args:
        text: Input text (cleaned/pre-processed).
        keywords: Extracted keyword set for classification hints.

    Returns:
        FullIntent with graph and constraints populated.
    """
    tokens = text.split()

    # Token-level classification into actions/objects/modifiers
    graph = extract_intent_graph(tokens, keywords)

    # Phrase-level constraint extraction (prepositional phrases)
    constraints = extract_constraints(text)

    return FullIntent(graph=graph, constraints=constraints)


# ═══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_from_graph(
    intent: FullIntent,
    include_modifiers: bool = False,
    max_constraints: int = 3,
) -> str:
    """
    Rebuild a compressed prompt from the intent graph.

    Produces a concise imperative sentence using only the
    essential semantic components:  action + objects [+ constraints].

    Args:
        intent: The extracted FullIntent.
        include_modifiers: If True, include top modifiers.
        max_constraints: Maximum number of constraints to include.

    Returns:
        Compressed prompt string.
    """
    d = intent.to_dict()

    parts: List[str] = []

    # Actions → imperative verb(s)
    actions = d["actions"]
    if actions:
        # Capitalise the first action for imperative tone
        parts.append(actions[0].capitalize())
        if len(actions) > 1:
            parts[-1] += " and " + ", ".join(actions[1:])

    # Objects → direct targets
    objects = d["objects"]
    if objects:
        parts.append(" ".join(objects[:8]))  # cap to avoid run-on

    # Modifiers (optional)
    if include_modifiers and d["modifiers"]:
        top_mods = d["modifiers"][:3]
        parts.append("(" + ", ".join(top_mods) + ")")

    # Constraints → prepositional phrases
    constraints = d["constraints"][:max_constraints]
    if constraints:
        parts.append("; ".join(constraints))

    prompt = " ".join(parts)
    return normalize_whitespace(prompt)


def compress_via_intent(
    text: str,
    keywords: Set[str],
    include_modifiers: bool = False,
    max_constraints: int = 3,
) -> Tuple[str, FullIntent]:
    """
    End-to-end: extract intent then reconstruct a compressed prompt.

    Returns:
        (compressed_prompt, full_intent)
    """
    intent = extract_full_intent(text, keywords)
    compressed = reconstruct_from_graph(
        intent,
        include_modifiers=include_modifiers,
        max_constraints=max_constraints,
    )
    return compressed, intent
