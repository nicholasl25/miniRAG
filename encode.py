from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer


class QueryEncoder:
    """Component 1: Encodes user queries into embedding vectors."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
    
    def encode(self, query: str) -> np.ndarray:
        """
        Convert a user query into an embedding vector.
        
        Args:
            query: Natural language question string
            
        Returns:
            numpy array with float32 dtype
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embedding.astype(np.float32)
