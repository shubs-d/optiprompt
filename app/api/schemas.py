"""Typed request/response contracts for API routes."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

Mode = Literal["aggressive", "balanced", "safe"]

class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw prompt text to optimize")
    mode: Mode = Field(default="balanced", description="Optimization profile")
    include_candidates: bool = Field(default=False)
    debug: bool = Field(default=False)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

class Metrics(BaseModel):
    semantic_similarity: float
    compression_gain: float
    original_token_count: Optional[int] = None
    optimized_token_count: Optional[int] = None

class OptimizeResponse(BaseModel):
    original_prompt: str
    optimized_prompt: str
    intent: str
    variants: List[str]
    token_reduction_percent: Optional[float] = None
    metrics: Metrics
