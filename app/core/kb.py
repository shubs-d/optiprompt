import json
import os
from typing import Dict, List, Set, Tuple

KB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "compression_kb.json")

class KnowledgeBase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeBase, cls).__new__(cls)
            cls._instance.load()
        return cls._instance

    def load(self):
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load KB from {KB_PATH}: {e}")
            self.data = []

        self.filler_phrases: List[str] = []
        self.phrase_replacements: List[Tuple[str, str]] = []
        self.conversational_tokens: Set[str] = set()
        self.low_value_modifiers: Set[str] = set()
        self.stopwords: Set[str] = set()

        for item in self.data:
            text = item.get("text", "")
            cat = item.get("category", "")
            action = item.get("action", "remove")
            if not text:
                continue
                
            if cat in ("filler_phrase", "framing", "politeness"):
                self.filler_phrases.append(text)
            elif cat == "phrase_replacement" and action == "replace":
                self.phrase_replacements.append((text, item.get("replacement", "")))
            elif cat == "filler_word":
                self.conversational_tokens.add(text)
            elif cat == "modifier":
                self.low_value_modifiers.add(text)
            elif cat == "connector":
                self.stopwords.add(text)

        # Ensure correct longest-first matching logic
        self.filler_phrases.sort(key=len, reverse=True)
        self.phrase_replacements.sort(key=lambda x: len(x[0]), reverse=True)

kb = KnowledgeBase()
