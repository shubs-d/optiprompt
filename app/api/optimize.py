"""FastAPI route for deterministic prompt optimization."""

from typing import Any, Dict, Literal, Optional
from datetime import datetime, timezone
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.cleaner import clean, normalize_text
from app.core.compressor import compress
from app.core.tokenizer import tokenize
from app.intent.classifier import detect_intent
from app.gepa.entropy import prune_prompt
from app.generation.generator import generate_variants
from app.semantic.similarity import semantic_similarity
from app.logging.logger import logger

router = APIRouter(prefix="/optimize", tags=["optimize"])

class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw prompt text to optimize")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Custom constraints like max_tokens")
    compute_mode: Literal["fast", "balanced", "aggressive"] = Field(
        default="balanced",
        description="Compute mode: fast skips LLM variant generation"
    )

class OptimizeResponse(BaseModel):
    optimized_prompt: str
    metrics: Dict[str, Any]
    system: Dict[str, Any]

@router.post("", response_model=OptimizeResponse)
def optimize_prompt(payload: OptimizeRequest) -> Dict[str, Any]:
    try:
        raw_prompt = payload.prompt.strip()
        normalized = normalize_text(raw_prompt)
        orig_tokens = len(tokenize(normalized)) or 1
        
        # 1. Constraint Parser
        constraints = payload.constraints
        min_tokens = constraints.get("min_tokens", 5)
        # Default to compressing to at least 80% if not specified
        max_tokens = constraints.get("max_tokens", max(10, int(orig_tokens * 0.8)))
        semantic_threshold = constraints.get("semantic_threshold", 0.85)
        
        # 2. Intent Detection
        intent_label = detect_intent(normalized, use_semantic_refinement=True)
            
        # 3. Aggression Controller Loop setup
        low, high = 0.0, 1.0
        aggression = 0.5
        max_iterations = 3
        
        best_candidate = None
        best_score = -100.0
        last_candidate = None
        
        for iteration in range(max_iterations):
            cleaned = clean(normalized, filler_strength=0.8)
            simplified = compress(cleaned, level=0.7)
            
            # GEPA (distilgpt2)
            pruned = prune_prompt(simplified, aggression=aggression)
            
            variants = {"gepa_baseline": pruned}
            
            # Variant Generation (Phi-2/TinyLlama)
            if payload.compute_mode != "fast":
                gen_variants = generate_variants(pruned, aggression=aggression)
                variants.update(gen_variants)
                
            iteration_best = None
            iteration_best_score = -100.0
            
            # Semantic Evaluation (MiniLM)
            for name, text in variants.items():
                sim = semantic_similarity(normalized, text)
                tok_count = len(tokenize(text)) or 1
                ratio = tok_count / orig_tokens
                
                # We want high similarity and high reduction (low ratio)
                score = 0.5 * sim + 0.5 * (1.0 - ratio)
                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best = {
                        "text": text,
                        "token_count": tok_count,
                        "semantic_similarity": sim,
                        "score": score,
                        "name": name
                    }
                    
            if not iteration_best:
                continue
                
            last_candidate = {**iteration_best, "aggression": aggression, "iterations": iteration + 1}
            tok_count = iteration_best["token_count"]
            sim = iteration_best["semantic_similarity"]
            
            is_valid = (min_tokens <= tok_count <= max_tokens) and (sim >= semantic_threshold)
            
            if is_valid and iteration_best_score > best_score:
                best_score = iteration_best_score
                best_candidate = {**iteration_best, "aggression": aggression, "iterations": iteration + 1}
            
            # Binary Search Adjustment
            if tok_count > max_tokens:
                low = aggression
            elif tok_count < min_tokens or sim < semantic_threshold:
                high = aggression
            else:
                if is_valid:
                    if best_candidate is None:
                        best_candidate = {**iteration_best, "aggression": aggression, "iterations": iteration + 1}
                    break
                    
            aggression = (low + high) / 2.0
            
        if not best_candidate:
            best_candidate = last_candidate or {
                "text": raw_prompt,
                "token_count": orig_tokens,
                "semantic_similarity": 1.0,
                "aggression": 0.0,
                "iterations": max_iterations,
                "name": "fallback"
            }
            
        opt_text = best_candidate["text"]
        opt_tokens = best_candidate["token_count"]
        # Make tokens accurate for extreme reduction
        if opt_tokens == 0: opt_tokens = 1
        
        token_reduction_percent = max(0.0, 1.0 - (opt_tokens / orig_tokens)) * 100.0
        compression_gain = max(0.0, 1.0 - (opt_tokens / orig_tokens))
        sim = float(best_candidate["semantic_similarity"])
        
        metrics = {
            "original_tokens": orig_tokens,
            "optimized_tokens": opt_tokens,
            "token_reduction_percent": round(token_reduction_percent, 2),
            "semantic_similarity": round(sim, 4),
            "compression_gain": round(compression_gain, 4)
        }
        
        models_used = {
            "gepa": "distilgpt2",
            "semantic": "MiniLM"
        }
        if payload.compute_mode != "fast":
            models_used["generation"] = "phi-2 | tinyllama"
            
        system = {
            "models_used": models_used,
            "compute_mode": payload.compute_mode,
            "final_aggression": round(best_candidate.get("aggression", 0.0), 4),
            "iterations": best_candidate.get("iterations", 1)
        }
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {
                "original_prompt": raw_prompt,
                "constraints": constraints
            },
            "output": {
                "optimized_prompt": opt_text,
                "intent": str(intent_label)
            },
            "metrics": metrics,
            "system": system
        }
        
        # Save JSON Log
        logger.log(log_entry)
        
        # Return Response
        return {
            "optimized_prompt": opt_text,
            "metrics": metrics,
            "system": system
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc
