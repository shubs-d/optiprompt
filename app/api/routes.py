"""Primary route layer for prompt optimization workflows."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException
import asyncio

from app.api.schemas import OptimizeRequest, OptimizeResponse
from app.core.pipeline import OptiPromptPipeline

router = APIRouter(tags=["optiprompt"])
pipeline = OptiPromptPipeline()

@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_prompt(payload: OptimizeRequest) -> Dict[str, Any]:
    try:
        # Run synchronous pipeline logic in a threadpool to not block the async event loop
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, pipeline.optimize, payload.prompt, payload)
        return raw
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc
