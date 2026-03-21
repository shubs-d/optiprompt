"""
text_utils.py — Shared text utilities for OptiPrompt.

Provides sentence splitting, word counting, n-gram generation,
stopword/filler sets, and other text helpers used across the pipeline.
"""

import re
from typing import List, Set, Tuple


# ── Stopwords (high-frequency, low-information words) ────────────────────────
STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "am", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their", "mine", "yours", "hers", "ours", "theirs", "what", "which",
    "who", "whom", "whose", "where", "when", "how", "if", "then", "else",
    "so", "but", "and", "or", "not", "no", "nor", "as", "at", "by", "for",
    "from", "in", "into", "of", "on", "to", "with", "about", "between",
    "through", "during", "before", "after", "above", "below", "up", "down",
    "out", "off", "over", "under", "again", "further", "than", "too",
    "very", "just", "also", "still", "already", "yet", "here", "there",
}

# ── Filler phrases (ordered longest-first for greedy matching) ───────────────
FILLER_PHRASES: List[str] = sorted([
    "please make sure to",
    "make sure to",
    "make sure that",
    "please ensure that",
    "please note that",
    "it is important to note that",
    "it should be noted that",
    "it is worth noting that",
    "it is worth mentioning that",
    "as a matter of fact",
    "as you can see",
    "as we all know",
    "at the end of the day",
    "basically",
    "essentially",
    "fundamentally",
    "in terms of",
    "kind of",
    "sort of",
    "more or less",
    "for the most part",
    "for all intents and purposes",
    "in my opinion",
    "in other words",
    "needless to say",
    "to be honest",
    "to tell you the truth",
    "truth be told",
    "at this point in time",
    "at the present time",
    "in the event that",
    "in the process of",
    "on a regular basis",
    "on a daily basis",
    "each and every",
    "first and foremost",
    "one and only",
    "point in time",
    "with respect to",
    "with regard to",
    "with regards to",
    "in regard to",
    "in regards to",
    "as a result of",
    "as a consequence of",
    "for the purpose of",
    "for the sake of",
    "by means of",
    "by virtue of",
    "in light of",
    "in view of",
    "on account of",
    "on behalf of",
    "in addition to",
    "in conjunction with",
    "in accordance with",
    "in compliance with",
    "pretty much",
    "quite frankly",
    "honestly speaking",
    "to be fair",
    "having said that",
    "that being said",
    "with that being said",
    "all things considered",
    "when all is said and done",
    "i would like you to",
    "i want you to",
    "i need you to",
    "you should try to",
    "you need to make sure",
    "please kindly",
    "kindly",
    "could you please",
    "would you please",
    "please be sure to",
    "be sure to",
    "don't forget to",
    "remember to",
], key=len, reverse=True)

# ── Phrase replacement map ───────────────────────────────────────────────────
PHRASE_REPLACEMENTS: List[Tuple[str, str]] = [
    ("in order to", "to"),
    ("due to the fact that", "because"),
    ("owing to the fact that", "because"),
    ("on the grounds that", "because"),
    ("for the reason that", "because"),
    ("by reason of", "because"),
    ("in spite of the fact that", "although"),
    ("despite the fact that", "although"),
    ("regardless of the fact that", "although"),
    ("notwithstanding the fact that", "although"),
    ("in the event that", "if"),
    ("on the condition that", "if"),
    ("assuming that", "if"),
    ("provided that", "if"),
    ("in the case that", "if"),
    ("in the case of", "for"),
    ("at this point in time", "now"),
    ("at the present time", "now"),
    ("at the current moment", "now"),
    ("at this moment", "now"),
    ("at this time", "now"),
    ("prior to", "before"),
    ("subsequent to", "after"),
    ("in the vicinity of", "near"),
    ("in close proximity to", "near"),
    ("a large number of", "many"),
    ("a great deal of", "much"),
    ("a significant amount of", "much"),
    ("a majority of", "most"),
    ("a small number of", "few"),
    ("in the near future", "soon"),
    ("until such time as", "until"),
    ("for the purpose of", "to"),
    ("with the exception of", "except"),
    ("in the absence of", "without"),
    ("in the presence of", "with"),
    ("in the amount of", "for"),
    ("on a daily basis", "daily"),
    ("on a regular basis", "regularly"),
    ("on a weekly basis", "weekly"),
    ("on a monthly basis", "monthly"),
    ("is able to", "can"),
    ("has the ability to", "can"),
    ("has the capacity to", "can"),
    ("be able to", "can"),
    ("whether or not", "whether"),
    ("the reason being", "because"),
    ("the reason is that", "because"),
    ("in such a manner as to", "to"),
    ("take into consideration", "consider"),
    ("take into account", "consider"),
    ("give consideration to", "consider"),
    ("make a decision", "decide"),
    ("come to a conclusion", "conclude"),
    ("reach a conclusion", "conclude"),
    ("conduct an investigation", "investigate"),
    ("perform an analysis", "analyze"),
    ("carry out", "do"),
    ("bring about", "cause"),
    ("put together", "assemble"),
    ("the fact that", "that"),
    ("it is important that", "importantly,"),
    ("it is necessary that", "must"),
    ("it is essential that", "must"),
    ("it is crucial that", "must"),
    ("there is a need to", "must"),
    ("has the potential to", "can"),
    ("is in a position to", "can"),
]

# Sort by length (longest first) for greedy replacement
PHRASE_REPLACEMENTS.sort(key=lambda x: len(x[0]), reverse=True)

# ── Low-value adjectives/adverbs ────────────────────────────────────────────
LOW_VALUE_MODIFIERS: Set[str] = {
    "very", "really", "quite", "rather", "somewhat", "fairly",
    "pretty", "extremely", "incredibly", "absolutely", "totally",
    "completely", "entirely", "utterly", "thoroughly", "perfectly",
    "highly", "deeply", "greatly", "strongly", "particularly",
    "especially", "specifically", "basically", "essentially",
    "literally", "actually", "definitely", "certainly", "obviously",
    "clearly", "simply", "merely", "just", "only", "even",
    "frankly", "honestly", "truly", "surely", "naturally",
    "apparently", "evidently", "presumably", "arguably",
    "undoubtedly", "unquestionably", "indisputably",
    "nice", "good", "great", "wonderful", "amazing", "awesome",
    "fantastic", "excellent", "outstanding", "remarkable",
    "significant", "substantial", "considerable", "notable",
    "careful", "carefully", "proper", "properly", "appropriate",
    "appropriately", "adequate", "adequately",
}


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex-based rules."""
    # Split on period/exclamation/question followed by space+uppercase or end
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in parts if s.strip()]


def count_words(text: str) -> int:
    """Count words in text (whitespace-split)."""
    return len(text.split())


def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def is_meaningful_token(token: str) -> bool:
    """Check if a token carries semantic meaning (not a stopword or punct)."""
    cleaned = token.lower().strip(".,!?;:\"'()-")
    if not cleaned:
        return False
    if cleaned in STOPWORDS:
        return False
    if len(cleaned) <= 1 and not cleaned.isalpha():
        return False
    return True


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single spaces."""
    return re.sub(r'\s+', ' ', text).strip()
