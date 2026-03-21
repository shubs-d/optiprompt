"""Token and cost estimation utilities."""

from typing import Dict

from app.core.tokenizer import count_tokens_simple, count_tokens_tiktoken

# Prices are in USD per 1K tokens (input-side approximation).
_MODEL_PRICES_PER_1K: Dict[str, float] = {
    "gpt-4": 0.03,
    "gpt-3.5": 0.0015,
}



def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count using tiktoken when available, with deterministic fallback."""
    tk_count = count_tokens_tiktoken(text, model=model)
    if tk_count is not None:
        return tk_count
    return count_tokens_simple(text)



def estimate_cost(text: str, model: str = "gpt-4") -> float:
    """Estimate prompt cost in USD."""
    tokens = estimate_tokens(text, model=model)
    per_1k = _MODEL_PRICES_PER_1K.get(model, _MODEL_PRICES_PER_1K["gpt-4"])
    return round((tokens / 1000.0) * per_1k, 8)



def estimate_cost_savings(original_text: str, optimized_text: str, model: str = "gpt-4") -> float:
    """Estimated USD saved by optimization for one prompt execution."""
    original_cost = estimate_cost(original_text, model=model)
    optimized_cost = estimate_cost(optimized_text, model=model)
    return round(max(0.0, original_cost - optimized_cost), 8)
