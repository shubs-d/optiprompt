"""Cosine similarity for semantic equivalence."""

import numpy as np
from app.semantic.embedding import get_embedding

def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0
        
    emb1 = np.array(get_embedding(text1))
    emb2 = np.array(get_embedding(text2))
    
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(emb1, emb2) / (norm1 * norm2))
