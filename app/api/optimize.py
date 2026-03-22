"""FastAPI route for deterministic prompt optimization."""

from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.pipeline import OptiPromptPipeline, PipelineConfig

router = APIRouter(prefix="/optimize", tags=["optimize"])
pipeline = OptiPromptPipeline()


class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw prompt text to optimize")
    mode: Literal["aggressive", "balanced", "safe"] = Field(
        default="balanced",
        description="Optimization profile",
    )
    include_candidates: bool = Field(
        default=False,
        description="Include scored intermediate candidates",
    )
    debug: bool = Field(default=False, description="Return stage-by-stage details")
    seed: int = Field(default=42, ge=0, le=2**31 - 1, description="Deterministic seed")
    rl_enabled: bool = Field(default=True, description="Enable RL strategy selection")
    rl_q_table_path: str = Field(
        default=".optiprompt_q_table.json",
        description="Path to persisted RL Q-table JSON",
    )
    rl_alpha: float = Field(default=0.2, ge=0.0, le=1.0, description="Q-learning rate")
    rl_gamma: float = Field(default=0.0, ge=0.0, le=1.0, description="Discount factor")
    rl_epsilon: float = Field(default=0.05, ge=0.0, le=1.0, description="Exploration rate")


class OptimizeResponse(BaseModel):
    original_prompt: str
    optimized_prompt: str
    compression_ratio: float
    token_reduction_percent: float
    keyword_retention: float
    information_density: float
    estimated_cost_savings: float
    redundancy_removed: float
    concepts_preserved: float
    agreement_score: float
    semantic_confidence_score: float
    metrics: Dict[str, Any]
    mode: str
    candidates: Optional[Any] = None
    debug: Optional[Any] = None


@router.post("", response_model=OptimizeResponse)
def optimize_prompt(payload: OptimizeRequest) -> Dict[str, Any]:
    try:
        cfg = PipelineConfig(
            mode=payload.mode,
            seed=payload.seed,
            include_candidates=payload.include_candidates,
            debug=payload.debug,
            rl_enabled=payload.rl_enabled,
            rl_q_table_path=payload.rl_q_table_path,
            rl_alpha=payload.rl_alpha,
            rl_gamma=payload.rl_gamma,
            rl_epsilon=payload.rl_epsilon,
        )
        return pipeline.optimize(payload.prompt, cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc
