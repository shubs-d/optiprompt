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
from app.structuring.transformer import extract_role, extract_objective, extract_constraints, extract_input, build_structured_prompt

router = APIRouter(prefix="/optimize", tags=["optimize"])

class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw prompt text to optimize")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Custom constraints like max_tokens")
    compute_mode: Literal["fast", "balanced", "high_quality"] = Field(
        default="balanced",
        description="Compute mode: fast skips LLM variant generation"
    )

class OptimizeResponse(BaseModel):
    structured_prompt: str
    compression_ratio: float
    structure_applied: bool
    optimized_prompt: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    system: Optional[Dict[str, Any]] = None

@router.post("", response_model=OptimizeResponse)
def optimize_prompt(payload: OptimizeRequest) -> Dict[str, Any]:
    try:
        raw_prompt = payload.prompt.strip()
        normalized = normalize_text(raw_prompt)
        orig_tokens = len(tokenize(normalized)) or 1
        
        # 1. Compute Mode Limits
        compute_mode = payload.compute_mode
        if compute_mode == "fast":
            MAX_COMPRESSION = 0.5
        elif compute_mode == "balanced":
            MAX_COMPRESSION = 0.4
        elif compute_mode == "high_quality":
            MAX_COMPRESSION = 0.3
        else:
            MAX_COMPRESSION = 0.4
            
        # 2. Constraints & Parameters
        TARGET_RANGE = (0.25, 0.45)
        MIN_SEMANTIC = 0.88
        
        # 3. Intent Detection
        intent_label = detect_intent(normalized, use_semantic_refinement=True)
            
        # 4. Semantic Structuring Logic
        words = raw_prompt.split()
        complexity_score = len(set(words)) / max(1, len(words))
        
        structure_applied = False
        working_prompt = normalized
        
        if len(words) > 20 or complexity_score > 0.8:
            role = extract_role(raw_prompt)
            objective = extract_objective(raw_prompt)
            req_constraints = extract_constraints(raw_prompt)
            extracted_input = extract_input(raw_prompt)
            
            working_prompt = build_structured_prompt(role, objective, req_constraints, extracted_input)
            structure_applied = True
            
            # Reduce GEPA aggression limit
            MAX_COMPRESSION = min(MAX_COMPRESSION, 0.4)
            
        # 5. Aggression Controller
        aggression = 0.5
        max_iterations = 3
        
        best_candidate = None
        best_score = -100.0
        
        fallback_candidate = {
            "text": working_prompt,
            "token_count": len(tokenize(working_prompt)) or 1,
            "semantic_similarity": 1.0,
            "aggression": 0.0,
            "iterations": 1,
            "compression_ratio": max(0.0, 1.0 - ((len(tokenize(working_prompt)) or 1) / orig_tokens)),
            "name": "fallback"
        }
        
        for iteration in range(max_iterations):
            if not structure_applied:
                cleaned = clean(working_prompt, filler_strength=0.8)
                simplified = compress(cleaned, level=0.7)
            else:
                simplified = working_prompt
                
            # GEPA Modification Protected Search
            pruned = prune_prompt(simplified, aggression=aggression)
            variants = {"gepa_baseline": pruned}
            
            if payload.compute_mode != "fast":
                gen_variants = generate_variants(pruned, aggression=aggression)
                variants.update(gen_variants)
                
            iteration_best = None
            iteration_best_score = -100.0
            
            # Semantic Evaluation & Updated Scoring
            for name, text in variants.items():
                sim = float(semantic_similarity(normalized, text))
                tok_count = len(tokenize(text)) or 1
                compression_ratio = max(0.0, 1.0 - (tok_count / orig_tokens))
                
                # Semantic Floor AND Hard Compression Cap valid tests 
                is_valid = (compression_ratio <= MAX_COMPRESSION) and (sim >= MIN_SEMANTIC)
                constraint_satisfaction = 1.0 if is_valid else 0.0
                
                # Built metrics
                structure_score = 0.5
                if "\n-" in text or "\n*" in text or "\n1." in text:
                    structure_score += 0.3
                if "structure" in name.lower():
                    structure_score += 0.1
                structure_score = min(1.0, structure_score)
                
                # Updated Score func execution
                score = (
                    0.5 * sim +
                    0.25 * (1.0 - compression_ratio) +
                    0.15 * structure_score +
                    0.10 * constraint_satisfaction
                )
                
                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best = {
                        "text": text,
                        "token_count": tok_count,
                        "semantic_similarity": sim,
                        "compression_ratio": compression_ratio,
                        "score": score,
                        "is_valid": is_valid,
                        "name": name
                    }
                    
            if not iteration_best:
                continue
                
            last_candidate = {**iteration_best, "aggression": aggression, "iterations": iteration + 1}
            
            if iteration_best["is_valid"] and iteration_best_score > best_score:
                best_score = iteration_best_score
                best_candidate = last_candidate
            
            # Target Compression Range adjustment
            comp_ratio = iteration_best["compression_ratio"]
            if comp_ratio > TARGET_RANGE[1]:
                aggression -= 0.1
            elif comp_ratio < TARGET_RANGE[0]:
                aggression += 0.1
                
            aggression = max(0.0, min(0.7, aggression))
            
        if not best_candidate:
            best_candidate = fallback_candidate
            
        opt_text = best_candidate["text"]
        opt_tokens = best_candidate["token_count"]
        token_reduction_percent = best_candidate["compression_ratio"] * 100.0
        compression_gain = best_candidate["compression_ratio"]
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
            "final_aggression": round(best_candidate.get("aggression", 0.0), 2),
            "iterations": best_candidate.get("iterations", 1)
        }
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {
                "original_prompt": raw_prompt,
                "constraints": payload.constraints
            },
            "output": {
                "optimized_prompt": opt_text,
                "intent": str(intent_label)
            },
            "metrics": metrics,
            "system": system
        }
        
        logger.log(log_entry)
        
        return {
            "structured_prompt": opt_text,
            "compression_ratio": round(compression_gain, 4),
            "structure_applied": structure_applied,
            "optimized_prompt": opt_text,
            "metrics": metrics,
            "system": system
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc
