"""
semantic.py — Stage 5: Semantic Compression Layer (Algorithmic).

Implements concept-level prompt compression using deterministic rules:
1. Intent graph extraction (actions, objects, constraints, modifiers)
2. Concept normalization via synonym clusters
3. Redundancy elimination (duplicate concept detection)
4. Phrase compression (verbose → concise transformations)
5. Dependency preservation (action-object integrity checks)
6. Concept-level pruning (remove low-value concepts)

No ML, no embeddings — fully deterministic.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from app.core.keyword_extractor import INSTRUCTION_VERBS, TECHNICAL_TERMS, NOUN_SUFFIXES
from app.utils.text_utils import (
    STOPWORDS,
    LOW_VALUE_MODIFIERS,
    normalize_whitespace,
    split_sentences,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SemanticConfig:
    """Controls semantic compression aggressiveness."""
    redundancy_threshold: float = 0.5      # 0.0–1.0: how aggressively to collapse synonyms
    modifier_pruning_level: float = 0.5    # 0.0–1.0: how aggressively to prune modifiers
    enable_synonym_collapse: bool = True
    enable_phrase_compression: bool = True
    enable_concept_pruning: bool = True


# Mode presets — mirrors pipeline mode profiles
SEMANTIC_MODE_CONFIGS: Dict[str, SemanticConfig] = {
    "aggressive": SemanticConfig(
        redundancy_threshold=0.8,
        modifier_pruning_level=0.9,
        enable_synonym_collapse=True,
        enable_phrase_compression=True,
        enable_concept_pruning=True,
    ),
    "balanced": SemanticConfig(
        redundancy_threshold=0.5,
        modifier_pruning_level=0.6,
        enable_synonym_collapse=True,
        enable_phrase_compression=True,
        enable_concept_pruning=True,
    ),
    "safe": SemanticConfig(
        redundancy_threshold=0.25,
        modifier_pruning_level=0.3,
        enable_synonym_collapse=True,
        enable_phrase_compression=True,
        enable_concept_pruning=False,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INTENT GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntentGraph:
    """Structured representation of prompt semantics."""
    actions: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)

    @property
    def total_concepts(self) -> int:
        return len(self.actions) + len(self.objects) + len(self.constraints) + len(self.modifiers)


# Preposition starters for constraint detection
_PREP_PATTERN = re.compile(
    r'\b(for|with|without|by|using|via|through|from|into|onto|upon|'
    r'against|between|among|across|during|before|after|until|since|'
    r'within|throughout|beyond|beneath|beside|towards?)\b\s+',
    re.IGNORECASE,
)

# Verb suffixes used for action detection (complements INSTRUCTION_VERBS)
_VERB_SUFFIXES = ("ify", "ize", "ise", "ate", "ing")


def extract_intent_graph(tokens: List[str], keywords: Set[str]) -> IntentGraph:
    """
    Convert a token list into a structured intent graph.

    Classification rules:
    - Tokens in INSTRUCTION_VERBS or with verb suffixes → actions
    - Tokens in TECHNICAL_TERMS, with noun suffixes, or capitalized → objects
    - Tokens in LOW_VALUE_MODIFIERS or common adjective suffixes → modifiers
    - Everything else meaningful → objects (default bucket)
    """
    graph = IntentGraph()

    for token in tokens:
        cleaned = token.lower().strip(".,!?;:\"'()-[]{}")
        if not cleaned or len(cleaned) < 2 or cleaned in STOPWORDS:
            continue

        # Actions: instruction verbs or verb-like morphology
        if cleaned in INSTRUCTION_VERBS:
            graph.actions.append(cleaned)
            continue

        if any(cleaned.endswith(s) for s in _VERB_SUFFIXES) and len(cleaned) > 3:
            # Check it's not a noun (e.g., "meeting", "building" as noun)
            if cleaned not in TECHNICAL_TERMS and not _has_noun_suffix(cleaned):
                graph.actions.append(cleaned)
                continue

        # Modifiers: low-value adjectives/adverbs or adjective-like suffixes
        if cleaned in LOW_VALUE_MODIFIERS:
            graph.modifiers.append(cleaned)
            continue
        if _has_adjective_suffix(cleaned) and cleaned not in keywords:
            graph.modifiers.append(cleaned)
            continue

        # Objects: technical terms, nouns, keywords
        if cleaned in TECHNICAL_TERMS or cleaned in keywords:
            graph.objects.append(cleaned)
            continue
        if _has_noun_suffix(cleaned) and len(cleaned) > 4:
            graph.objects.append(cleaned)
            continue

        # Capitalized tokens → likely entities/objects
        if token[0].isupper() and cleaned.isalpha() and len(cleaned) > 2:
            graph.objects.append(cleaned)
            continue

        # Remaining meaningful tokens → objects (conservative)
        if len(cleaned) > 3 and cleaned.isalpha():
            graph.objects.append(cleaned)

    return graph


def extract_constraints(text: str) -> List[str]:
    """Extract prepositional-phrase constraints from text."""
    constraints: List[str] = []
    for match in _PREP_PATTERN.finditer(text):
        start = match.start()
        # Grab the phrase up to the next comma, period, or end
        rest = text[start:]
        end_match = re.search(r'[,.\n;]', rest)
        phrase = rest[:end_match.start()] if end_match else rest
        phrase = phrase.strip()
        if 3 < len(phrase.split()) <= 12:
            constraints.append(phrase)
    return constraints


def _has_noun_suffix(word: str) -> bool:
    return any(word.endswith(s) for s in NOUN_SUFFIXES)


def _has_adjective_suffix(word: str) -> bool:
    adj_suffixes = ("ful", "less", "ous", "ive", "able", "ible", "ical", "ish")
    return any(word.endswith(s) for s in adj_suffixes)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONCEPT NORMALIZATION — Synonym Clusters
# ═══════════════════════════════════════════════════════════════════════════════

# Each cluster is a frozenset. The representative is chosen as:
# shortest word, breaking ties alphabetically.
SYNONYM_CLUSTERS: List[frozenset] = [
    # Speed
    frozenset({"fast", "quick", "efficient", "rapid", "swift", "speedy"}),
    # Creation
    frozenset({"build", "create", "develop", "construct", "make", "produce"}),
    # Explanation
    frozenset({"explain", "describe", "elaborate", "illustrate", "clarify"}),
    # Analysis
    frozenset({"analyze", "examine", "evaluate", "assess", "inspect", "review"}),
    # Improvement
    frozenset({"improve", "enhance", "optimize", "refine", "upgrade", "boost"}),
    # Size/Scope (large)
    frozenset({"large", "big", "huge", "enormous", "vast", "massive", "extensive"}),
    # Size/Scope (small)
    frozenset({"small", "little", "tiny", "minor", "minimal", "slight"}),
    # Detail
    frozenset({"detailed", "comprehensive", "thorough", "extensive", "in-depth", "exhaustive"}),
    # Generation
    frozenset({"generate", "produce", "create", "output", "yield"}),
    # Display
    frozenset({"show", "display", "present", "render", "exhibit"}),
    # Implementation
    frozenset({"implement", "execute", "deploy", "run", "perform", "carry out"}),
    # Documentation
    frozenset({"document", "record", "log", "chronicle", "note"}),
    # Testing
    frozenset({"test", "verify", "validate", "check", "confirm"}),
    # Removal
    frozenset({"remove", "delete", "eliminate", "discard", "drop"}),
    # Configuration
    frozenset({"configure", "setup", "set up", "initialize", "establish"}),
    # Acquisition
    frozenset({"get", "obtain", "acquire", "fetch", "retrieve", "gather"}),
    # Sending
    frozenset({"send", "transmit", "dispatch", "deliver", "forward"}),
    # Importance
    frozenset({"important", "crucial", "critical", "essential", "vital", "key"}),
    # Accuracy
    frozenset({"accurate", "precise", "exact", "correct", "right"}),
    # Simplicity
    frozenset({"simple", "easy", "straightforward", "basic", "uncomplicated"}),
    # Speed modifiers
    frozenset({"quickly", "rapidly", "swiftly", "promptly", "speedily", "hastily"}),
    # Certainty modifiers
    frozenset({"definitely", "certainly", "surely", "undoubtedly", "absolutely"}),
    # Completeness modifiers
    frozenset({"completely", "entirely", "fully", "totally", "wholly"}),
    # Clarity
    frozenset({"clear", "obvious", "apparent", "evident", "plain"}),
    # Quality
    frozenset({"good", "great", "excellent", "outstanding", "superior", "fine"}),
    # Difficulty
    frozenset({"difficult", "hard", "challenging", "complex", "tough"}),
    # Reliability
    frozenset({"reliable", "dependable", "stable", "robust", "resilient"}),
    # Flexibility
    frozenset({"flexible", "adaptable", "versatile", "modular", "adjustable"}),
    # Communication
    frozenset({"communicate", "convey", "relay", "inform", "notify"}),
    # Storage
    frozenset({"store", "save", "persist", "cache", "retain"}),
]

# Pre-compute lookup: word → (cluster_index, representative)
_WORD_TO_CLUSTER: Dict[str, Tuple[int, str]] = {}
_CLUSTER_REPS: Dict[int, str] = {}

def _init_cluster_lookup() -> None:
    """Build the word → cluster lookup table."""
    for idx, cluster in enumerate(SYNONYM_CLUSTERS):
        # Representative: shortest word, then alphabetically first
        rep = sorted(cluster, key=lambda w: (len(w), w))[0]
        _CLUSTER_REPS[idx] = rep
        for word in cluster:
            _WORD_TO_CLUSTER[word] = (idx, rep)

_init_cluster_lookup()


def normalize_concepts(tokens: List[str], threshold: float = 0.5) -> Tuple[List[str], int]:
    """
    Replace synonym-cluster words with their representative.

    Args:
        tokens: Word-level tokens.
        threshold: 0.0–1.0 controlling aggressiveness. Below 0.3, only
                   collapse when 3+ synonyms from the same cluster appear.

    Returns:
        (normalized_tokens, num_replacements)
    """
    if threshold <= 0.0:
        return tokens, 0

    # Count how many tokens fall in each cluster
    cluster_counts: Dict[int, int] = {}
    for token in tokens:
        cleaned = token.lower().strip(".,!?;:\"'()-")
        if cleaned in _WORD_TO_CLUSTER:
            cidx, _ = _WORD_TO_CLUSTER[cleaned]
            cluster_counts[cidx] = cluster_counts.get(cidx, 0) + 1

    # Determine which clusters to collapse based on threshold
    # Low threshold → only collapse clusters with many duplicates
    min_occurrences = max(2, int(4 * (1.0 - threshold)))

    active_clusters: Set[int] = {
        cidx for cidx, count in cluster_counts.items()
        if count >= min_occurrences
    }

    # At high thresholds (≥0.6), also activate any cluster with ≥2 occurrences
    if threshold >= 0.6:
        active_clusters.update(
            cidx for cidx, count in cluster_counts.items() if count >= 2
        )

    # Replace tokens
    result: List[str] = []
    seen_clusters: Set[int] = set()
    replacements = 0

    for token in tokens:
        cleaned = token.lower().strip(".,!?;:\"'()-")
        if cleaned in _WORD_TO_CLUSTER:
            cidx, rep = _WORD_TO_CLUSTER[cleaned]
            if cidx in active_clusters:
                if cidx not in seen_clusters:
                    # First occurrence of this cluster → use representative
                    seen_clusters.add(cidx)
                    # Preserve original casing style
                    if token[0].isupper():
                        rep = rep.capitalize()
                    result.append(rep)
                else:
                    # Duplicate from same cluster → skip
                    replacements += 1
                    continue
            else:
                result.append(token)
        else:
            result.append(token)

    return result, replacements


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REDUNDANCY ELIMINATION
# ═══════════════════════════════════════════════════════════════════════════════

# Common redundant adjective pairs (keeping only the first element)
_REDUNDANT_PAIRS: List[Tuple[str, str]] = [
    ("detailed", "comprehensive"),
    ("clear", "concise"),
    ("fast", "efficient"),
    ("fast", "quick"),
    ("complete", "thorough"),
    ("robust", "reliable"),
    ("secure", "safe"),
    ("simple", "easy"),
    ("new", "novel"),
    ("important", "critical"),
    ("unique", "distinct"),
    ("brief", "concise"),
    ("accurate", "precise"),
    ("modern", "contemporary"),
    ("flexible", "adaptable"),
    ("scalable", "extensible"),
]

# Redundant phrase patterns (regex → keep-version)
_REDUNDANT_PHRASE_PATTERNS: List[Tuple[str, str]] = [
    # "X and Y" where X ≈ Y
    (r'\bdetailed\s+and\s+comprehensive\b', 'detailed'),
    (r'\bcomprehensive\s+and\s+detailed\b', 'comprehensive'),
    (r'\bclear\s+and\s+concise\b', 'clear'),
    (r'\bconcise\s+and\s+clear\b', 'concise'),
    (r'\bfast\s+and\s+efficient\b', 'efficient'),
    (r'\befficient\s+and\s+fast\b', 'efficient'),
    (r'\bfast\s+and\s+quick\b', 'fast'),
    (r'\bquick\s+and\s+fast\b', 'fast'),
    (r'\bfast\s+and\s+efficient\s+and\s+quick\b', 'fast'),
    (r'\bcomplete\s+and\s+thorough\b', 'thorough'),
    (r'\brobust\s+and\s+reliable\b', 'robust'),
    (r'\bsecure\s+and\s+safe\b', 'secure'),
    (r'\bsimple\s+and\s+easy\b', 'simple'),
    (r'\baccurate\s+and\s+precise\b', 'accurate'),
    (r'\bflexible\s+and\s+adaptable\b', 'flexible'),
    (r'\bscalable\s+and\s+extensible\b', 'scalable'),
    # "X, Y, and Z" redundant triplets
    (r'\bdetailed\s*,\s*comprehensive\s*,?\s*and\s+thorough\b', 'detailed'),
    (r'\bfast\s*,\s*quick\s*,?\s*and\s+efficient\b', 'fast'),
    (r'\bclear\s*,\s*concise\s*,?\s*and\s+brief\b', 'clear'),
]


def eliminate_redundancy(text: str) -> Tuple[str, float]:
    """
    Remove redundant semantic pairs/triplets from text.

    Returns:
        (cleaned_text, redundancy_score) where redundancy_score is
        the fraction of duplicate concepts that were eliminated.
    """
    original_word_count = len(text.split())
    eliminations = 0

    # Apply redundant phrase patterns
    for pattern, replacement in _REDUNDANT_PHRASE_PATTERNS:
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new_text != text:
            eliminations += 1
            text = new_text

    # Detect same-cluster adjacency not caught by patterns
    words = text.split()
    cleaned_words: List[str] = []
    prev_cluster: Optional[int] = None

    for word in words:
        cleaned = word.lower().strip(".,!?;:\"'()-")
        if cleaned in _WORD_TO_CLUSTER:
            cidx, _ = _WORD_TO_CLUSTER[cleaned]
            if cidx == prev_cluster:
                # Same cluster as previous word → skip duplicate
                eliminations += 1
                continue
            prev_cluster = cidx
        else:
            prev_cluster = None
        cleaned_words.append(word)

    result = ' '.join(cleaned_words)
    new_word_count = len(result.split())

    # redundancy_score = duplicate_concepts / total_concepts
    redundancy_score = eliminations / max(1, original_word_count)
    return result, round(redundancy_score, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PHRASE COMPRESSION (EXTENDED)
# ═══════════════════════════════════════════════════════════════════════════════

# Additional verbose → concise transformations beyond text_utils.PHRASE_REPLACEMENTS
_SEMANTIC_PHRASE_MAP: List[Tuple[str, str]] = sorted([
    # Structural verbosity
    ("a system that is able to", "system to"),
    ("a tool that can be used to", "tool to"),
    ("a process that involves", "process of"),
    ("a method that allows you to", "method to"),
    ("an approach that enables", "approach enabling"),
    ("a way to be able to", "way to"),
    ("a solution that provides", "solution providing"),
    ("the ability to be able to", "ability to"),
    # Hedging / softening
    ("it might be worth considering", "consider"),
    ("you may want to consider", "consider"),
    ("it would be beneficial to", ""),
    ("it is recommended that you", ""),
    ("it is advisable to", ""),
    ("it is suggested that", ""),
    ("it is generally accepted that", ""),
    ("it can be argued that", ""),
    ("one could say that", ""),
    ("it is widely known that", ""),
    # Verbose connectors
    ("in addition to the above", "also"),
    ("on top of that", "also"),
    ("as well as the fact that", "and"),
    ("not only that but also", "and"),
    ("along with the fact that", "and"),
    # Wordy instruction patterns
    ("please ensure that you", ""),
    ("you should always make sure to", ""),
    ("it is important to always", "always"),
    ("you must always remember to", "always"),
    ("do not forget to always", "always"),
    ("be careful to make sure that", "ensure"),
    ("take care to ensure that", "ensure"),
    # Common padding (safe to remove entirely)
    ("as mentioned previously", ""),
    ("as stated earlier", ""),
    ("as noted above", ""),
    ("as described below", ""),
    ("it is worth pointing out that", ""),
    ("it bears mentioning that", ""),
    ("the thing is that", ""),
    ("what this means is that", ""),
], key=lambda x: len(x[0]), reverse=True)


def compress_phrases(text: str) -> str:
    """Apply deterministic verbose → concise phrase replacements."""
    for verbose, concise in _SEMANTIC_PHRASE_MAP:
        pattern = re.compile(re.escape(verbose), re.IGNORECASE)
        text = pattern.sub(concise, text)
    # Clean up double spaces and leading/trailing artifacts
    text = normalize_whitespace(text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DEPENDENCY PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════════

def verify_dependencies(
    original_graph: IntentGraph,
    compressed_text: str,
    original_text: str,
) -> str:
    """
    Post-compression safety check. If critical semantic links are broken,
    fall back to the original text for that portion.

    Rules:
    - Every action must still have ≥1 object in the compressed text
    - If all actions are lost, restore original
    """
    compressed_lower = compressed_text.lower()

    # Check: do we still have at least one action?
    retained_actions = [a for a in original_graph.actions if a in compressed_lower]
    if not retained_actions and original_graph.actions:
        # Catastrophic loss — fall back to original
        return original_text

    # Check: does every retained action have ≥1 co-occurring object?
    retained_objects = [o for o in original_graph.objects if o in compressed_lower]
    if retained_actions and not retained_objects:
        # Actions exist but no objects — partial restoration
        # Append the top 3 original objects
        top_objects = list(dict.fromkeys(original_graph.objects))[:3]
        suffix = ", ".join(top_objects)
        return f"{compressed_text} ({suffix})"

    return compressed_text


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONCEPT-LEVEL PRUNING
# ═══════════════════════════════════════════════════════════════════════════════

def prune_low_value_concepts(
    text: str,
    graph: IntentGraph,
    keywords: Set[str],
    pruning_level: float = 0.5,
) -> Tuple[str, int]:
    """
    Remove low-value concepts from text while keeping semantic core.

    Removes:
    - Redundant modifiers (beyond the first per sentence)
    - Repeated intent signals (e.g., "please" appearing multiple times)
    - Filler descriptions that don't add instruction value

    Keeps:
    - All actions
    - Core objects (keywords, technical terms)
    - Essential constraints

    Args:
        text: Input text.
        graph: Extracted intent graph.
        keywords: Keywords to always preserve.
        pruning_level: 0.0–1.0 aggressiveness.

    Returns:
        (pruned_text, num_concepts_pruned)
    """
    if pruning_level <= 0.0:
        return text, 0

    pruned_count = 0
    sentences = split_sentences(text)
    result_sentences: List[str] = []

    # Set of modifier words to consider pruning
    prunable_modifiers = set(graph.modifiers)

    for sent in sentences:
        words = sent.split()
        kept_words: List[str] = []
        modifier_count_in_sent = 0
        # How many modifiers to keep per sentence (based on pruning level)
        max_modifiers = max(1, int(3 * (1.0 - pruning_level)))

        for word in words:
            cleaned = word.lower().strip(".,!?;:\"'()-")

            # Always keep keywords, actions, and objects
            if cleaned in keywords or cleaned in INSTRUCTION_VERBS or cleaned in TECHNICAL_TERMS:
                kept_words.append(word)
                continue

            # Prune excess modifiers
            if cleaned in prunable_modifiers:
                modifier_count_in_sent += 1
                if modifier_count_in_sent > max_modifiers:
                    pruned_count += 1
                    continue

            kept_words.append(word)

        result_sentences.append(' '.join(kept_words))

    # Remove duplicate "please" and other repeated intent signals
    result_text = ' '.join(result_sentences)
    # Keep only first "please"
    please_count = 0
    final_words = []
    for word in result_text.split():
        if word.lower().strip(".,!?") == "please":
            please_count += 1
            if please_count > 1:
                pruned_count += 1
                continue
        final_words.append(word)

    return ' '.join(final_words), pruned_count


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SemanticMetrics:
    """Metrics produced by the semantic compression layer."""
    redundancy_removed: float = 0.0       # fraction of duplicate concepts eliminated
    concepts_preserved: float = 1.0       # retained_concepts / original_concepts
    original_concept_count: int = 0
    retained_concept_count: int = 0
    synonym_replacements: int = 0
    concepts_pruned: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "redundancy_removed": round(self.redundancy_removed, 4),
            "concepts_preserved": round(self.concepts_preserved, 4),
            "original_concept_count": self.original_concept_count,
            "retained_concept_count": self.retained_concept_count,
            "synonym_replacements": self.synonym_replacements,
            "concepts_pruned": self.concepts_pruned,
        }


def semantic_compress(
    text: str,
    keywords: Set[str],
    mode: str = "balanced",
    config: Optional[SemanticConfig] = None,
) -> Tuple[str, SemanticMetrics]:
    """
    Apply the full semantic compression pipeline.

    Args:
        text: Pre-processed text (post rule-based pruning).
        keywords: Extracted keyword set.
        mode: 'aggressive', 'balanced', or 'safe'.
        config: Optional override for semantic config.

    Returns:
        (compressed_text, semantic_metrics)
    """
    if config is None:
        config = SEMANTIC_MODE_CONFIGS.get(mode, SEMANTIC_MODE_CONFIGS["balanced"])

    tokens = text.split()
    metrics = SemanticMetrics()

    # ── Step 1: Extract intent graph ─────────────────────────────────────
    graph = extract_intent_graph(tokens, keywords)
    metrics.original_concept_count = graph.total_concepts

    # ── Step 2: Concept normalization (synonym collapsing) ───────────────
    if config.enable_synonym_collapse:
        tokens, synonym_count = normalize_concepts(
            tokens, threshold=config.redundancy_threshold
        )
        metrics.synonym_replacements = synonym_count
        text = ' '.join(tokens)

    # ── Step 3: Redundancy elimination ───────────────────────────────────
    text, redundancy_score = eliminate_redundancy(text)
    metrics.redundancy_removed = redundancy_score

    # ── Step 4: Phrase compression ───────────────────────────────────────
    if config.enable_phrase_compression:
        text = compress_phrases(text)

    # ── Step 5: Dependency preservation ──────────────────────────────────
    text = verify_dependencies(graph, text, ' '.join(tokens))

    # ── Step 6: Concept-level pruning ────────────────────────────────────
    if config.enable_concept_pruning:
        text, pruned = prune_low_value_concepts(
            text, graph, keywords,
            pruning_level=config.modifier_pruning_level,
        )
        metrics.concepts_pruned = pruned

    # ── Final: calculate preserved concepts ──────────────────────────────
    final_tokens = text.split()
    final_graph = extract_intent_graph(final_tokens, keywords)
    metrics.retained_concept_count = final_graph.total_concepts
    metrics.concepts_preserved = (
        final_graph.total_concepts / max(1, metrics.original_concept_count)
    )

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text, metrics
