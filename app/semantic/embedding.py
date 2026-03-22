"""Generate text embeddings using MiniLM."""

from app.models.loader import model_loader
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1024)
def get_embedding(text: str) -> tuple:
    """Get embedding for text. Cached to prevent recomputation."""
    model = model_loader.get_minilm()
    embedding = model.encode(text)
    return tuple(embedding.tolist())
