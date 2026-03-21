"""
test_semantic_optimization.py — Tests for the new semantic optimization modules.

Tests:
1. intent_graph.py — FullIntent extraction and reconstruction
2. semantic_rules.py — Semantic deduplication engine
3. critic.py — Textual critic feedback loop
4. formatter.py — Markdown output formatter
5. Full pipeline integration (11-stage)
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Intent Graph Extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_intent_graph_extraction():
    from app.core.intent_graph import extract_full_intent, reconstruct_from_graph

    text = "Create a detailed API document using JSON format for the backend team"
    keywords = {"api", "json", "backend", "document"}

    intent = extract_full_intent(text, keywords)
    d = intent.to_dict()

    print("=== INTENT GRAPH ===")
    print(f"  Actions:     {d['actions']}")
    print(f"  Objects:     {d['objects']}")
    print(f"  Constraints: {d['constraints']}")
    print(f"  Modifiers:   {d['modifiers']}")

    assert len(d["actions"]) > 0, "Should extract at least one action"
    assert len(d["objects"]) > 0, "Should extract at least one object"

    # Test reconstruction
    compressed = reconstruct_from_graph(intent)
    print(f"  Reconstructed: {compressed}")
    assert len(compressed) > 0, "Reconstruction should produce non-empty text"
    assert len(compressed) < len(text), "Reconstruction should be shorter than original"

    print("  ✓ test_intent_graph_extraction PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Semantic Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def test_semantic_deduplication():
    from app.core.semantic_rules import semantic_deduplicate

    text = (
        "Create a very extremely detailed report. "
        "Build a comprehensive and thorough analysis. "
        "Develop a detailed API document for the backend."
    )
    keywords = {"api", "report", "analysis", "backend"}

    deduped, metrics = semantic_deduplicate(text, keywords, aggressiveness=0.6)

    print("=== SEMANTIC DEDUPLICATION ===")
    print(f"  Original:  {text}")
    print(f"  Deduped:   {deduped}")
    print(f"  Concepts removed: {metrics.duplicate_concepts_removed}")
    print(f"  Modifier chains:  {metrics.redundant_modifier_chains}")
    print(f"  Sentences merged: {metrics.similar_sentences_merged}")
    print(f"  Redundancy score: {metrics.total_redundancy_score:.4f}")

    assert len(deduped.split()) <= len(text.split()), "Deduplication should not increase word count"
    assert metrics.total_redundancy_score >= 0.0, "Redundancy score should be non-negative"

    print("  ✓ test_semantic_deduplication PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Textual Critic
# ─────────────────────────────────────────────────────────────────────────────

def test_critic_over_compression():
    from app.core.critic import refine

    original = "Please create a detailed API design document that explains all endpoints and includes examples for each one"
    over_compressed = "API doc."  # way too short
    keywords = {"api", "document", "endpoints", "examples"}
    instructions = {"create", "explain", "include"}

    refined, report = refine(over_compressed, original, keywords, instructions)

    print("=== TEXTUAL CRITIC (Over-compression) ===")
    print(f"  Input:    {over_compressed}")
    print(f"  Refined:  {refined}")
    print(f"  Report:   {report.to_dict()}")

    # The critic should detect and partially fix the over-compression
    assert len(refined.split()) > len(over_compressed.split()), "Critic should add back content"
    assert report.passes_applied >= 1, "Should apply at least one pass"

    print("  ✓ test_critic_over_compression PASSED\n")


def test_critic_healthy_candidate():
    from app.core.critic import refine

    original = "Create a detailed API design document that explains all endpoints"
    healthy = "Create detailed API design document explaining all endpoints"
    keywords = {"api", "document", "endpoints"}
    instructions = {"create", "explain"}

    refined, report = refine(healthy, original, keywords, instructions)

    print("=== TEXTUAL CRITIC (Healthy candidate) ===")
    print(f"  Input:    {healthy}")
    print(f"  Refined:  {refined}")
    print(f"  Report:   {report.to_dict()}")

    # Healthy candidate should pass through mostly unchanged
    assert report.passes_applied >= 1, "Should run at least one pass"

    print("  ✓ test_critic_healthy_candidate PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Markdown Formatter
# ─────────────────────────────────────────────────────────────────────────────

def test_markdown_formatter():
    from app.core.formatter import format_as_markdown, format_compact

    text = "Create a detailed API document using JSON format for the backend team"
    keywords = {"api", "json", "backend", "document"}

    md = format_as_markdown(text, keywords=keywords)
    compact = format_compact(text, keywords=keywords)

    print("=== MARKDOWN FORMATTER ===")
    print(f"  Full Markdown:\n{md}")
    print(f"\n  Compact: {compact}")

    assert "### Task" in md or "### Target" in md, "Should contain markdown sections"
    assert len(compact) > 0, "Compact format should be non-empty"

    print("  ✓ test_markdown_formatter PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Self-Consistency Selection
# ─────────────────────────────────────────────────────────────────────────────

def test_self_consistency():
    from app.core.transformations import compute_agreement_scores, select_by_consensus

    candidates = [
        {"name": "a", "text": "Create API document with endpoints", "fitness": 0.7},
        {"name": "b", "text": "Create API document including endpoints examples", "fitness": 0.8},
        {"name": "c", "text": "API document endpoints examples format", "fitness": 0.6},
    ]

    scores = compute_agreement_scores(candidates)

    print("=== SELF-CONSISTENCY SELECTION ===")
    for cand, score in zip(candidates, scores):
        print(f"  {cand['name']}: agreement={score:.4f}, fitness={cand['fitness']}")

    best, best_agreement = select_by_consensus(candidates)
    print(f"  Best: {best['name']} (agreement={best_agreement:.4f})")

    assert len(scores) == 3, "Should return score for each candidate"
    assert all(0.0 <= s <= 1.0 for s in scores), "Scores should be in [0, 1]"

    print("  ✓ test_self_consistency PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Full Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline():
    from app.core.pipeline import OptiPromptPipeline, PipelineConfig

    prompts = [
        "Please make sure to create a detailed and comprehensive API design document "
        "that explains all endpoints clearly and concisely and also includes examples "
        "for each one in order to help the development team implement it efficiently.",

        "I would like you to basically analyze this very detailed report and provide "
        "a clear and concise summary that is very thorough and also includes actionable "
        "next steps for improving response times and reducing latency.",

        "The system should be able to generate an optimization plan due to the fact that "
        "token costs are too high and we need a compact, structured prompt format that "
        "is fast and efficient and quick to process.",
    ]

    pipeline = OptiPromptPipeline()

    print("=== FULL PIPELINE INTEGRATION ===")
    for i, prompt in enumerate(prompts):
        cfg = PipelineConfig(mode="balanced", seed=42)
        result = pipeline.optimize(prompt, cfg)

        compression = result["token_reduction_percent"]
        agreement = result["metrics"]["agreement_score"]
        concepts = result["concepts_preserved"]
        redundancy = result["redundancy_removed"]
        confidence = result["semantic_confidence_score"]

        print(f"\n  Prompt {i+1}:")
        print(f"    Original ({len(prompt.split())} words):  {prompt[:80]}...")
        print(f"    Optimized ({len(result['optimized_prompt'].split())} words): {result['optimized_prompt'][:80]}...")
        print(f"    Compression:     {compression:.1f}%")
        print(f"    Agreement:       {agreement:.4f}")
        print(f"    Concepts:        {concepts:.4f}")
        print(f"    Redundancy:      {redundancy:.4f}")
        print(f"    Confidence:      {confidence:.4f}")

        # Validate ranges
        assert 0.0 <= compression <= 80.0, f"Compression should be reasonable, got {compression}"
        assert 0.0 <= agreement <= 1.0, f"Agreement should be in [0, 1], got {agreement}"
        assert 0.0 <= concepts <= 1.0, f"Concepts should be in [0, 1], got {concepts}"
        assert 0.0 <= redundancy <= 1.0, f"Redundancy should be in [0, 1], got {redundancy}"
        assert 0.0 <= confidence <= 1.0, f"Confidence should be in [0, 1], got {confidence}"

        # Check new metric fields exist
        assert "agreement_score" in result["metrics"], "agreement_score missing from metrics"
        assert "agreement_score" in result, "agreement_score missing from top-level result"

    print("\n  ✓ test_full_pipeline PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  OptiPrompt Semantic Optimization Test Suite")
    print("=" * 60 + "\n")

    test_intent_graph_extraction()
    test_semantic_deduplication()
    test_critic_over_compression()
    test_critic_healthy_candidate()
    test_markdown_formatter()
    test_self_consistency()
    test_full_pipeline()

    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
