"""
test_normalization_spellcheck.py — Tests for conversational token elimination
and deterministic spell checking features.

Tests:
1. Elongation normalization
2. Slang normalization
3. Conversational token removal
4. Unified normalize_text()
5. Spell correction (word-level)
6. Spell correction (text-level)
7. End-to-end pipeline integration with spec example
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Elongation Normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_elongation():
    from app.core.cleaner import normalize_elongation

    cases = [
        ("heyyyy", "heyy"),
        ("goooood", "good"),
        ("hellllo", "hello"),     # 4 l's → 2 l's
        ("plzzzzz", "plzz"),
        ("normal", "normal"),     # no change
        ("aaabbb", "aabb"),       # both groups collapsed
    ]

    print("=== ELONGATION NORMALIZATION ===")
    for input_text, expected in cases:
        result = normalize_elongation(input_text)
        print(f"  '{input_text}' → '{result}' (expected: '{expected}')")
        assert result == expected, f"Expected '{expected}', got '{result}'"

    print("  ✓ test_normalize_elongation PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Slang Normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_slang():
    from app.core.cleaner import normalize_slang

    cases = [
        ("plz help me", "please help me"),
        ("thx for the info", "thanks for the information"),
        ("pls do this", "please do this"),
        ("normal words here", "normal words here"),
    ]

    print("=== SLANG NORMALIZATION ===")
    for input_text, expected in cases:
        result = normalize_slang(input_text)
        print(f"  '{input_text}' → '{result}' (expected: '{expected}')")
        assert result == expected, f"Expected '{expected}', got '{result}'"

    print("  ✓ test_normalize_slang PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Conversational Token Removal
# ─────────────────────────────────────────────────────────────────────────────

def test_remove_conversational_tokens():
    from app.core.cleaner import remove_conversational_tokens

    cases = [
        ("hello please help me build", "help me build"),
        ("hi thanks for helping", "for helping"),
        ("make it fast and efficient", "make it fast and efficient"),  # context words preserved
        ("hey bye", ""),  # all conversational
        ("build a secure api", "build a secure api"),  # no conversational tokens
    ]

    print("=== CONVERSATIONAL TOKEN REMOVAL ===")
    for input_text, expected in cases:
        result = remove_conversational_tokens(input_text)
        print(f"  '{input_text}' → '{result}' (expected: '{expected}')")
        assert result == expected, f"Expected '{expected}', got '{result}'"

    print("  ✓ test_remove_conversational_tokens PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Unified normalize_text()
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_text():
    from app.core.cleaner import normalize_text

    # The spec example
    input_text = "heyyyy plzzzz help me build a goooood aplication"
    result = normalize_text(input_text)

    print("=== UNIFIED normalize_text() ===")
    print(f"  Input:  '{input_text}'")
    print(f"  Output: '{result}'")

    # After normalization: "help me build good aplication"
    # - "heyyyy" → "heyy" (elongation) → removed (conversational token)
    # - "plzzzz" → "plzz" (elongation) → "please" (slang) → removed (conversational token)
    # - "goooood" → "good" (elongation)
    # - all lowercased
    assert "heyy" not in result, "heyy should be removed as conversational token"
    assert "please" not in result, "please should be removed as conversational token"
    assert "build" in result, "'build' should be preserved"
    assert "good" in result, "'goooood' should normalize to 'good'"

    print("  ✓ test_normalize_text PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Spell Correction (word-level)
# ─────────────────────────────────────────────────────────────────────────────

def test_correct_word():
    from app.core.spellcheck import correct_word

    cases = [
        ("aplication", "application"),
        ("bild", "build"),
        ("optimzation", "optimization"),
        ("api", "api"),             # already correct
        ("xyz123", "xyz123"),       # no match → return as-is
        ("build", "build"),         # exact match
    ]

    print("=== SPELL CORRECTION (word-level) ===")
    for input_word, expected in cases:
        result = correct_word(input_word)
        print(f"  '{input_word}' → '{result}' (expected: '{expected}')")
        assert result == expected, f"Expected '{expected}', got '{result}'"

    print("  ✓ test_correct_word PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Spell Correction (text-level)
# ─────────────────────────────────────────────────────────────────────────────

def test_spell_check_text():
    from app.core.spellcheck import spell_check_text

    cases = [
        ("build good aplication", "build good application"),
        ("optimze the systm", "optimize the system"),
        ("fast and efficent", "fast and efficient"),
    ]

    print("=== SPELL CORRECTION (text-level) ===")
    for input_text, expected in cases:
        result = spell_check_text(input_text)
        print(f"  '{input_text}' → '{result}' (expected: '{expected}')")
        assert result == expected, f"Expected '{expected}', got '{result}'"

    print("  ✓ test_spell_check_text PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: End-to-end Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_integration():
    from app.core.pipeline import OptiPromptPipeline, PipelineConfig

    prompt = "Heyyyy plzzzz help me build a goooood aplication that is fast and efficient"

    pipeline = OptiPromptPipeline()
    config = PipelineConfig(mode="balanced", seed=42)
    result = pipeline.optimize(prompt, config)

    optimized = result["optimized_prompt"]

    print("=== PIPELINE INTEGRATION ===")
    print(f"  Input:     '{prompt}'")
    print(f"  Optimized: '{optimized}'")
    print(f"  Compression: {result['token_reduction_percent']:.1f}%")

    # The output should not contain raw conversational tokens or misspellings
    lower_opt = optimized.lower()
    assert "heyy" not in lower_opt, "Conversational tokens should be removed"
    assert "plz" not in lower_opt, "Slang should be normalized and removed"
    assert "aplication" not in lower_opt, "Misspellings should be corrected"

    print("  ✓ test_pipeline_integration PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  OptiPrompt Normalization & Spell Check Test Suite")
    print("=" * 60 + "\n")

    test_normalize_elongation()
    test_normalize_slang()
    test_remove_conversational_tokens()
    test_normalize_text()
    test_correct_word()
    test_spell_check_text()
    test_pipeline_integration()

    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
