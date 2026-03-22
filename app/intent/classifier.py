"""Intent classification using rules and optional MiniLM semantic fallback."""

from app.semantic.similarity import semantic_similarity

INTENT_RULES = {
    "coding": ["code", "bug", "python", "function", "script", "debug"],
    "creative": ["story", "write", "imagine", "create", "poem"],
    "qa": ["what", "why", "how", "explain", "who", "when"],
}

INTENT_PROTOTYPES = {
    "coding": "write a python script to fix the code bug",
    "creative": "write a creative story and imagine a poem",
    "qa": "explain how and why what works",
    "general": "general text prompt",
}

def detect_intent(prompt: str, use_semantic_refinement: bool = True) -> str:
    """Detect intent using rule-based keywords, then optional semantic refinement."""
    prompt_lower = prompt.lower()
    
    # Step 1: Rule-based detection
    detected = {}
    for intent, keywords in INTENT_RULES.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            detected[intent] = score
            
    # Best rule-based intent
    best_intent = "general"
    if detected:
        best_intent = max(detected.items(), key=lambda x: x[1])[0]
        
    # Step 2: Semantic refinement (use embeddings vs predefined intent vectors)
    if use_semantic_refinement:
        semantic_scores = {}
        for intent, prototype in INTENT_PROTOTYPES.items():
            sim = semantic_similarity(prompt, prototype)
            semantic_scores[intent] = sim
            
        semantic_best = max(semantic_scores.items(), key=lambda x: x[1])
        # If semantic score is highly confident, override rule-based
        if semantic_best[1] > 0.4:
            best_intent = semantic_best[0]
            
    return best_intent
