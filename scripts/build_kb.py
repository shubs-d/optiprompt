import json
import os
import sys

# Ensure we can import the app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.rules import CONVERSATIONAL_TOKENS, LOW_VALUE_MODIFIERS
from app.utils.text_utils import FILLER_PHRASES, PHRASE_REPLACEMENTS, STOPWORDS

def get_category_and_priority(text: str, default_cat: str) -> tuple:
    text_len = len(text.split())
    if "please" in text or "kindly" in text or "thanks" in text or "appreciate" in text:
        return "politeness", 2
    if text.startswith("i want") or text.startswith("can you") or text.startswith("could you") or text.startswith("would you") or text.startswith("i need") or text.startswith("i would like"):
        return "framing", 3
    
    if default_cat == "phrase_replacement":
        return "phrase_replacement", 3
    if default_cat == "modifier":
        return "modifier", 2
    if default_cat == "connector":
        return "connector", 1

    if text_len > 1:
        return "filler_phrase", 2
    return "filler_word", 1

all_entries = []

for phrase in FILLER_PHRASES:
    cat, prio = get_category_and_priority(phrase, "filler_phrase")
    all_entries.append({
        "category": cat,
        "text": phrase,
        "action": "remove",
        "replacement": "",
        "priority": prio,
        "context_sensitive": False,
        "source": "extracted"
    })

for old, new in PHRASE_REPLACEMENTS:
    all_entries.append({
        "category": "phrase_replacement",
        "text": old,
        "action": "replace",
        "replacement": new,
        "priority": 3,
        "context_sensitive": False,
        "source": "extracted"
    })

for token in CONVERSATIONAL_TOKENS:
    cat, prio = get_category_and_priority(token, "filler_word")
    all_entries.append({
        "category": cat,
        "text": token,
        "action": "remove",
        "replacement": "",
        "priority": prio,
        "context_sensitive": False,
        "source": "extracted"
    })

for mod in LOW_VALUE_MODIFIERS:
    all_entries.append({
        "category": "modifier",
        "text": mod,
        "action": "remove",
        "replacement": "",
        "priority": 2,
        "context_sensitive": True,
        "source": "extracted"
    })

for word in STOPWORDS:
    all_entries.append({
        "category": "connector",
        "text": word,
        "action": "remove",
        "replacement": "",
        "priority": 1,
        "context_sensitive": True,
        "source": "extracted"
    })

# Deduplicate based on text
unique_entries = {}
for entry in all_entries:
    t = entry["text"]
    if t not in unique_entries:
        unique_entries[t] = entry
    else:
        # Prefer higher priority and 'replace' action
        current = unique_entries[t]
        if entry["action"] == "replace" and current["action"] == "remove":
            unique_entries[t] = entry
        elif entry["priority"] > current["priority"]:
            unique_entries[t] = entry

final_list = list(unique_entries.values())
# Sort by priority desc, then text len desc mapping back to original logic behavior
final_list.sort(key=lambda x: (-x['priority'], -len(x['text']), x['text']))

with open("app/data/compression_kb.json", "w") as f:
    json.dump(final_list, f, indent=2)

print(f"Exported {len(final_list)} unique entries to app/data/compression_kb.json")
