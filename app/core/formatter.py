"""
formatter.py — Markdown-Based Output Formatter.

Formats an optimised prompt into a structured markdown representation,
organised by semantic role (Task, Target, Constraints).

Optimised for token efficiency — no JSON/XML overhead.
"""

from typing import Dict, List, Optional, Set

from app.core.intent_graph import FullIntent, extract_full_intent


def format_as_markdown(
    text: str,
    intent: Optional[FullIntent] = None,
    keywords: Optional[Set[str]] = None,
) -> str:
    """
    Format an optimised prompt into markdown sections.

    Sections:
    - ### Task        — action verbs (what to do)
    - ### Target      — objects/entities (what to act on)
    - ### Constraints — prepositional-phrase constraints (how/when/where)

    If no intent is provided, extracts one from the text.

    Args:
        text: Optimised prompt text.
        intent: Pre-extracted FullIntent (optional).
        keywords: Keyword set for extraction (required if intent is None).

    Returns:
        Markdown-formatted prompt string.
    """
    if intent is None:
        keywords = keywords or set()
        intent = extract_full_intent(text, keywords)

    d = intent.to_dict()

    sections: List[str] = []

    # Task section: actions
    actions = d.get("actions", [])
    if actions:
        # Capitalise and join as imperative verbs
        action_str = ", ".join(a.capitalize() for a in actions)
        sections.append(f"### Task\n{action_str}")

    # Target section: objects
    objects = d.get("objects", [])
    if objects:
        object_str = ", ".join(objects)
        sections.append(f"### Target\n{object_str}")

    # Constraints section: prepositional phrases
    constraints = d.get("constraints", [])
    if constraints:
        constraint_lines = "\n".join(f"- {c}" for c in constraints)
        sections.append(f"### Constraints\n{constraint_lines}")

    # If we couldn't extract structured sections, fall back to raw text
    if not sections:
        return text

    return "\n\n".join(sections)


def format_compact(
    text: str,
    intent: Optional[FullIntent] = None,
    keywords: Optional[Set[str]] = None,
) -> str:
    """
    Format as a compact single-line structured prompt.

    Format: "ACTION: [verbs] | TARGET: [objects] | CONSTRAINTS: [phrases]"

    More token-efficient than full markdown for API usage.
    """
    if intent is None:
        keywords = keywords or set()
        intent = extract_full_intent(text, keywords)

    d = intent.to_dict()
    parts: List[str] = []

    actions = d.get("actions", [])
    if actions:
        parts.append(f"ACTION: {', '.join(actions)}")

    objects = d.get("objects", [])
    if objects:
        parts.append(f"TARGET: {', '.join(objects[:6])}")

    constraints = d.get("constraints", [])
    if constraints:
        parts.append(f"CONSTRAINTS: {'; '.join(constraints[:3])}")

    return " | ".join(parts) if parts else text
