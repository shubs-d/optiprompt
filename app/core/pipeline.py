"""Multi-Model 8-stage optimization pipeline for OptiPrompt."""

from typing import Dict, Any
from app.core.cleaner import clean, normalize_text
from app.core.compressor import compress
from app.core.tokenizer import tokenize
from app.intent.classifier import detect_intent
from app.gepa.entropy import calculate_surprisal
from app.generation.generator import generate_variants
from app.semantic.similarity import semantic_similarity

class OptiPromptPipeline:
    """Production-facing multi-model prompt optimization pipeline."""

    def __init__(self, **kwargs) -> None:
        pass

    def optimize(self, prompt: str, config: Any = None) -> Dict:
        raw_prompt = prompt.strip()
        if not raw_prompt:
            raise ValueError("Prompt must not be empty.")

        # Pre-normalization
        normalized_prompt = normalize_text(raw_prompt)
        original_token_count = len(tokenize(normalized_prompt))
        
        if original_token_count == 0:
            original_token_count = 1  # prevent div by zero
            
        # 1. Regex Cleaner
        cleaned = clean(normalized_prompt, filler_strength=0.8)
        
        # 2. Structural Simplifier
        simplified = compress(cleaned, level=0.7)
        
        # 3. Intent Detection
        intent = detect_intent(simplified, use_semantic_refinement=True)
        
        # 4. GEPA - Entropy calculation
        surprisal_scores_data = calculate_surprisal(simplified)
        surprisal_scores = [t[2] for t in surprisal_scores_data]
        avg_surprisal = sum(surprisal_scores) / len(surprisal_scores) if surprisal_scores else 0.0
        
        # 5. Variant Generation
        generated_variants = generate_variants(simplified)
        variants_dict = generated_variants.copy()
        variants_dict["structural_baseline"] = simplified
        
        # 6. Semantic Eval & 7. Select Best Variant
        best_variant = None
        best_score = -100.0
        scored_variants = []
        
        for name, text in variants_dict.items():
            sim = semantic_similarity(normalized_prompt, text)
            
            var_tokens = len(tokenize(text))
            compression_gain = max(0.0, 1.0 - (var_tokens / original_token_count))
            
            structure_score = 0.5
            if "\n-" in text or "\n*" in text or "\n1." in text:
                structure_score += 0.3
            if name == "structure-enhanced":
                structure_score += 0.1
                
            structure_score = min(1.0, structure_score)
            
            score = 0.5 * sim + 0.3 * compression_gain + 0.2 * structure_score
            
            scored_variants.append({
                "name": name,
                "text": text,
                "score": score,
                "semantic_similarity": sim,
                "compression_gain": compression_gain,
                "structure_score": structure_score
            })
            
            if score > best_score:
                best_score = score
                best_variant = scored_variants[-1]
                
        final_optimized = best_variant["text"] if best_variant else simplified
        final_sim = best_variant["semantic_similarity"] if best_variant else 0.0
        final_comp = best_variant["compression_gain"] if best_variant else 0.0
        
        optimized_token_count = len(tokenize(final_optimized))
        token_reduction_percent = max(0.0, 1.0 - (optimized_token_count / original_token_count)) * 100.0

        result = {
            "original_prompt": raw_prompt,
            "optimized_prompt": final_optimized,
            "intent": intent,
            "variants": [v["text"] for v in scored_variants if v["name"] != "structural_baseline"],
            "token_reduction_percent": token_reduction_percent,
            "metrics": {
                "semantic_similarity": final_sim,
                "compression_gain": final_comp,
                "original_token_count": original_token_count,
                "optimized_token_count": optimized_token_count
            }
        }
        
        return result
