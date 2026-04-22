# src/embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from src.config import settings
import json

class Embedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        # Load SentenceTransformer model (CPU/GPU auto-detect)
        self.model = SentenceTransformer(self.model_name)
        
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Convert texts to normalized embeddings.
        Returns: np.ndarray of shape (len(texts), embedding_dim)
        """
        # Use self.model.encode() with batch_size and show_progress_bar
        # Important: Set normalize_embeddings=True for cosine similarity (FAISS IP)
        embedding = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=show_progress)
        return embedding
    
    def encode_file(self, chunks_file: str, output_path: str, batch_size: int = 32):
        # Collect all texts (do this efficiently)
        texts = []
        with open(chunks_file, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                text = data['text']
                texts.append(text if text else None)
        
        print(f"Collected {len(texts)} texts. Starting encoding...")
        
        # Encode all at once (this uses the batch_size properly)
        embeddings = self.encode(texts, batch_size=batch_size, show_progress=True)
        
        # Save once
        np.save(output_path, embeddings)
        print(f"Embeddings saved. Shape: {embeddings.shape}")