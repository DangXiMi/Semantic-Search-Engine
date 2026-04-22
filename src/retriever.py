# src/retriever.py
from typing import List, Dict, Any
import numpy as np
from src.embedder import Embedder
from src.indexer import FAISSIndex
from src.config import settings

class Retriever:
    def __init__(self):
        # Initialize embedder (loads model)
        self.embedder = Embedder()
        
        # Initialize index and load from disk
        self.index = FAISSIndex()
        self.index.load(settings.INDEX_FILE, settings.METADATA_FILE)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute semantic search and return formatted results.
        Returns list of dicts with keys: id, title, text, score
        """
        # Encode query to normalized vector (float32)
        query_vec = self.embedder.encode([query], show_progress=False)
        query_vec = query_vec.astype(np.float32)
        
        # Search index (scores, metadata list)
        scores, metadata_list = self.index.search(query_vec, k=k)
        
        # Format combined results
        results = []
        for score, meta in zip(scores, metadata_list):
            results.append({
                "id": meta["id"],
                "title": meta["title"],
                "text": meta["text"],
                "score": float(score)  
            })
        return results