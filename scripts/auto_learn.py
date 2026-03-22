#!/usr/bin/env python3
"""
auto_learn.py — Offline script to process failed optimizations and update the KB.
Usage: python scripts/auto_learn.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.learning_engine import LearningEngine

def main():
    log_path = os.environ.get("FAILED_OPT_LOG_PATH")
    if not log_path:
        log_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'data', 'failed_optimizations.jsonl')
    
    if not os.path.exists(log_path):
        print(f"No failed optimization log found at {log_path}. Nothing to learn.")
        return

    engine = LearningEngine()
    if not engine.enabled:
        print("Learning Engine is not enabled (missing openai package or OPENAI_API_KEY).")
        return

    print("Reading failed optimization logs...")
    failures = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                failures.append(json.loads(line))
                
    if not failures:
        print("Log is empty. Nothing to learn.")
        return

    print(f"Processing {len(failures)} failed optimization(s)...")
    total_added = 0
    
    for fail in failures:
        new_entries = engine.analyze_failed_optimization(
            original=fail.get("original", ""),
            optimized=fail.get("optimized", ""),
            reward=fail.get("reward", 0.0)
        )
        if new_entries:
            added = engine.update_knowledge_base(new_entries)
            print(f"Added {added} new rule(s) from optimization (Reward {fail.get('reward', 0.0):.2f})")
            total_added += added
        else:
            print(f"No new rules extracted from optimization (Reward {fail.get('reward', 0.0):.2f})")
            
    print(f"\nAuto-learn completed. Extracted a total of {total_added} new linguistic rules into the KB.")
    
    # Optionally clear the log after processing so we don't re-process
    with open(log_path, 'w', encoding='utf-8') as f:
        pass
    print("Cleared failed_optimizations.jsonl")

if __name__ == "__main__":
    main()
