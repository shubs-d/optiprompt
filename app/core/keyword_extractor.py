"""
keyword_extractor.py — Stage 3: Rule-based Keyword Extraction.

Uses suffix/pattern heuristics and predefined word sets to identify
keywords (nouns, verbs, technical terms) and instruction verbs.
No ML, no POS tagging libraries.
"""

import re
from typing import List, Set, Tuple


# ── Predefined instruction / action verbs ────────────────────────────────────
INSTRUCTION_VERBS: Set[str] = {
    # Creation
    "generate", "create", "build", "make", "produce", "construct",
    "develop", "design", "compose", "draft", "write", "author",
    # Analysis
    "analyze", "evaluate", "assess", "examine", "inspect", "review",
    "audit", "investigate", "study", "research", "explore",
    # Communication
    "explain", "describe", "summarize", "outline", "list", "enumerate",
    "define", "clarify", "elaborate", "illustrate", "present",
    # Action
    "implement", "execute", "run", "deploy", "install", "configure",
    "setup", "initialize", "start", "stop", "restart", "update",
    "upgrade", "migrate", "convert", "transform", "translate",
    # Data operations
    "extract", "parse", "filter", "sort", "merge", "split", "join",
    "aggregate", "compute", "calculate", "count", "measure",
    # Modification
    "modify", "edit", "change", "adjust", "fix", "repair", "debug",
    "refactor", "optimize", "improve", "enhance", "extend",
    # Removal / restriction
    "remove", "delete", "drop", "exclude", "ignore", "skip",
    "disable", "restrict", "limit", "constrain", "prevent",
    # Comparison
    "compare", "contrast", "differentiate", "distinguish",
    # Output
    "return", "output", "display", "show", "print", "render",
    "format", "export", "save", "store", "log", "report",
    # Testing
    "test", "verify", "validate", "check", "confirm", "ensure",
    # Instruction
    "use", "apply", "include", "add", "set", "assign", "map",
    "specify", "provide", "supply", "give", "send", "receive",
}

# ── Noun-indicating suffixes ─────────────────────────────────────────────────
NOUN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ence", "ance",
    "ism", "ist", "ure", "dom", "ship", "hood", "ery",
    "ology", "ics", "phy", "thy",
)

# ── Technical / domain terms (always keep) ───────────────────────────────────
TECHNICAL_TERMS: Set[str] = {
    "api", "sql", "json", "xml", "html", "css", "http", "https",
    "rest", "graphql", "grpc", "tcp", "udp", "ssh", "ftp",
    "aws", "gcp", "azure", "docker", "kubernetes", "k8s",
    "database", "schema", "table", "query", "index", "cache",
    "server", "client", "endpoint", "webhook", "socket",
    "token", "auth", "oauth", "jwt", "session", "cookie",
    "frontend", "backend", "fullstack", "microservice",
    "algorithm", "function", "class", "module", "package",
    "variable", "parameter", "argument", "config", "configuration",
    "deployment", "pipeline", "workflow", "process", "thread",
    "memory", "cpu", "gpu", "disk", "network", "latency",
    "error", "exception", "bug", "issue", "debug", "log",
    "test", "unittest", "integration", "benchmark", "performance",
    "security", "encryption", "firewall", "proxy", "gateway",
    "code", "script", "binary", "library", "framework",
    "repository", "branch", "commit", "merge", "release",
    "python", "javascript", "typescript", "java", "golang",
    "react", "vue", "angular", "node", "express", "fastapi",
    "linux", "windows", "macos", "ubuntu", "debian",
    "report", "system", "architecture", "component", "model",
    "data", "input", "output", "result", "response", "request",
}


def extract_keywords(tokens: List[str]) -> Tuple[Set[str], Set[str]]:
    """
    Extract keywords and instruction verbs from tokenized text.

    Returns:
        Tuple of (keyword_set, instruction_set):
        - keyword_set: all semantically important tokens (nouns, technical terms)
        - instruction_set: action/instruction verbs found
    """
    keyword_set: Set[str] = set()
    instruction_set: Set[str] = set()

    for token in tokens:
        cleaned = token.lower().strip(".,!?;:\"'()-[]{}").strip()
        if not cleaned or len(cleaned) < 2:
            continue

        # Check instruction verbs
        if cleaned in INSTRUCTION_VERBS:
            instruction_set.add(cleaned)
            keyword_set.add(cleaned)
            continue

        # Check technical terms
        if cleaned in TECHNICAL_TERMS:
            keyword_set.add(cleaned)
            continue

        # Check noun suffixes
        if _has_noun_suffix(cleaned) and len(cleaned) > 4:
            keyword_set.add(cleaned)
            continue

        # Check for capitalized terms (likely proper nouns / entities)
        if token[0].isupper() and len(cleaned) > 2 and cleaned.isalpha():
            keyword_set.add(cleaned)
            continue

        # Check for ALL CAPS (likely acronyms)
        if token.isupper() and len(token) >= 2 and token.isalpha():
            keyword_set.add(cleaned)
            continue

        # Check for camelCase or PascalCase (code identifiers)
        if _is_camel_case(token):
            keyword_set.add(cleaned)
            continue

        # Check for hyphenated compound terms
        if '-' in token and len(cleaned) > 4:
            keyword_set.add(cleaned)
            continue

        # Verb-like patterns: words ending in common verb suffixes
        if _has_verb_suffix(cleaned) and len(cleaned) > 3:
            keyword_set.add(cleaned)
            continue

    return keyword_set, instruction_set


def _has_noun_suffix(word: str) -> bool:
    """Check if a word ends with a common noun suffix."""
    return any(word.endswith(suffix) for suffix in NOUN_SUFFIXES)


def _has_verb_suffix(word: str) -> bool:
    """Check if a word has verb-like morphology."""
    verb_patterns = ("ify", "ize", "ise", "ate")
    return any(word.endswith(p) for p in verb_patterns)


def _is_camel_case(word: str) -> bool:
    """Check if word is camelCase or PascalCase."""
    return bool(re.match(r'^[a-z]+[A-Z]', word) or
                re.match(r'^[A-Z][a-z]+[A-Z]', word))
