import json
import os
import re
from typing import List, Dict, Any
from app.core.kb import kb, KB_PATH

LEARNING_PROMPT_TEMPLATE = """You are a Prompt Optimization Learning Engine.
Your task is to improve a JSON knowledge base by identifying missed filler words or phrases from failed optimizations.

---
## 📥 INPUT
Original Prompt:
{original}

Optimized Prompt:
{optimized}

Reward Score:
{reward}

---
## 🎯 OBJECTIVE
If the reward score is LOW (< 0.6), identify:
* filler words not removed
* redundant phrases still present
* missed compression opportunities

## 🧠 DETECTION RULES
Look for:
* conversational noise (you know, I mean, etc.)
* redundant phrasing (in order to, due to the fact that)
* weak modifiers (very, really)
* instruction wrappers (can you, I want you to)

## ⚙️ OUTPUT NEW ENTRIES
Return ONLY new entries not already in KB.

## 📤 OUTPUT FORMAT (STRICT JSON ARRAY)
[
{{
  "category": "<category>",
  "text": "<detected_phrase>",
  "action": "remove",
  "replacement": "",
  "priority": 1,
  "context_sensitive": true,
  "source": "auto_learned"
}}
]

## ⚠️ RULES
* Do NOT duplicate existing entries
* Only include meaningful additions
* Keep phrases short (1–5 words)
* Return STRICTLY a valid JSON array and nothing else.
"""

class LearningEngine:
    def __init__(self):
        # We attempt to import openai here to keep it optional for standard OptiPrompt usage
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.enabled = True
        except ImportError:
            print("Warning: openai package not installed. Learning engine disabled.")
            self.enabled = False
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client ({e}). Learning engine disabled.")
            self.enabled = False

    def analyze_failed_optimization(self, original: str, optimized: str, reward: float) -> List[Dict[str, Any]]:
        """Invokes the LLM to learn new mapping rules from failures."""
        if not self.enabled or reward >= 0.6:
            return []

        prompt = LEARNING_PROMPT_TEMPLATE.format(
            original=original,
            optimized=optimized,
            reward=reward
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content or "[]"
            return self._parse_json_array(content)
        except Exception as e:
            print(f"LLM API Error during learning phase: {e}")
            return []

    def _parse_json_array(self, content: str) -> List[Dict[str, Any]]:
        """Extracts JSON array safely from an LLM response markdown block."""
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        return []

    def update_knowledge_base(self, new_entries: List[Dict[str, Any]]) -> int:
        """Appends new valid LLM-extracted rules to the KB json safely."""
        if not new_entries:
            return 0
            
        with open(KB_PATH, "r", encoding="utf-8") as f:
            current_data = json.load(f)
            
        existing_texts = {item.get("text", "").lower() for item in current_data}
        added_count = 0
        
        for entry in new_entries:
            text = entry.get("text", "").lower()
            if text and text not in existing_texts:
                if entry.get("source") != "auto_learned":
                    entry["source"] = "auto_learned"
                current_data.append(entry)
                existing_texts.add(text)
                added_count += 1
                
        if added_count > 0:
            with open(KB_PATH, "w", encoding="utf-8") as f:
                json.dump(current_data, f, indent=2)
            # Re-initialize the KB singleton in memory
            kb.load()
            
        return added_count
