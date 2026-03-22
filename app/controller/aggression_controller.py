"""Dynamic Aggression Controller using Binary Search optimization."""

from typing import Dict, Tuple
from app.core.cleaner import clean, normalize_text
from app.core.compressor import compress
from app.core.tokenizer import tokenize
from app.gepa.entropy import prune_prompt
from app.generation.generator import generate_variants
from app.semantic.similarity import semantic_similarity

class AggressionController:
    """Feedback-driven optimization system to compress prompts within specific token limits."""

    def __init__(
        self, 
        target_token_range: Tuple[int, int], 
        semantic_threshold: float = 0.85, 
        max_iterations: int = 5
    ):
        self.min_tokens, self.max_tokens = target_token_range
        self.semantic_threshold = semantic_threshold
        self.max_iterations = max_iterations

    def optimize(self, prompt: str) -> Dict:
        """Dynamically tune compression aggression to fit target bounds."""
        
        normalized = normalize_text(prompt)
        original_tokens = len(tokenize(normalized))
        if original_tokens == 0:
            original_tokens = 1
            
        # Binary search state
        low = 0.0
        high = 1.0
        aggression = 0.5
        
        best_candidate = None
        best_score = -100.0
        
        last_iteration_best = None
        
        for iteration in range(self.max_iterations):
            # Run compression pipeline with current aggression
            cleaned = clean(normalized, filler_strength=0.8)
            simplified = compress(cleaned, level=0.7)
            pruned = prune_prompt(simplified, aggression=aggression)
            
            variants = generate_variants(pruned, aggression=aggression)
            variants["structural_baseline"] = pruned
            
            iteration_best = None
            iteration_best_score = -100.0
            
            # Semantic evaluation constraint checking
            for name, text in variants.items():
                sim = semantic_similarity(normalized, text)
                tok_count = len(tokenize(text))
                ratio = tok_count / original_tokens
                
                # Best score formulation: 0.5 * semantic + 0.5 * (1 - token_ratio)
                score = 0.5 * sim + 0.5 * (1.0 - ratio)
                
                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best = {
                        "text": text,
                        "token_count": tok_count,
                        "semantic_similarity": sim,
                        "score": score
                    }
                    
            if not iteration_best:
                continue
                
            last_iteration_best = iteration_best
            tok_count = iteration_best["token_count"]
            sim = iteration_best["semantic_similarity"]
            
            is_valid = (self.min_tokens <= tok_count <= self.max_tokens) and (sim >= self.semantic_threshold)
            
            if is_valid and iteration_best_score > best_score:
                best_score = iteration_best_score
                best_candidate = {
                    **iteration_best, 
                    "aggression": aggression, 
                    "iterations": iteration + 1
                }
                
            # Feedback-driven Binary Search Adjustment
            if tok_count > self.max_tokens:
                # Prompt too long -> need more compression
                low = aggression
            elif tok_count < self.min_tokens or sim < self.semantic_threshold:
                # Prompt too short or meaning lost -> need less compression
                high = aggression
            else:
                # Perfect range match and valid semantic preservation
                if is_valid:
                    # If we found a valid candidate right in the middle, we can stop early
                    break
                    
            aggression = (low + high) / 2.0
            
        # Fallback to the last available iteration if no constraints were met perfectly
        if best_candidate is None and last_iteration_best is not None:
            best_candidate = {
                **last_iteration_best, 
                "aggression": aggression, 
                "iterations": self.max_iterations
            }
            
        if not best_candidate:
             best_candidate = {
                 "text": prompt, 
                 "token_count": original_tokens, 
                 "semantic_similarity": 1.0, 
                 "aggression": 0.0, 
                 "iterations": 1
             }

        return {
            "optimized_prompt": best_candidate["text"],
            "final_token_count": best_candidate["token_count"],
            "target_range": [self.min_tokens, self.max_tokens],
            "semantic_similarity": best_candidate["semantic_similarity"],
            "final_aggression": best_candidate["aggression"],
            "iterations_used": best_candidate["iterations"]
        }

if __name__ == "__main__":
    # Example execution
    prompt_example = "Could you please write a detailed and comprehensive guide on how to efficiently deploy machine learning models on AWS using Kubernetes, including step by step tutorials for beginners?"
    
    print("Initializing Controller...")
    # Set targets: between 10 and 20 tokens, min 0.85 similarity
    controller = AggressionController(
        target_token_range=(10, 20),
        semantic_threshold=0.85,
        max_iterations=5
    )
    
    print(f"Original Prompt: {prompt_example}")
    result = controller.optimize(prompt_example)
    
    import json
    print("\n--- RESULTS ---")
    print(json.dumps(result, indent=2))
