# src/indexer.py
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
from src.config import settings

class FAISSIndex:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.metadata: List[Dict] = []  # list of dicts, index = FAISS ID

    def build(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """
        Build a FAISS index from embeddings and store metadata.
        embeddings: (N, D) normalized vectors for cosine similarity.
        """
        assert embeddings.shape[1] == self.dimension
        
        # Create faiss.IndexFlatIP(dimension)
        index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings (ensure they are float32 contiguous)
        index.add(embeddings)
        
        # Store metadata in self.metadata
        self.index = index
        self.metadata = metadata
        

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Persist index and metadata to disk."""
        # faiss.write_index(self.index, str(index_path))
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        

    def load(self, index_path: Path, metadata_path: Path) -> None:
        """Load index and metadata from disk."""
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index or metadata missing at {index_path.parent}")
        
        self.index = faiss.read_index(str(index_path))
        self.metadata = pickle.load(open(metadata_path, 'rb'))
        

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[float], List[Dict]]:
        """
        Search for k nearest neighbors.
        query_vector: (1, D) or (D,) array, normalized.
        Returns: (scores, metadata_list)
        """
        query = np.asarray(query_vector, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)

        assert query.shape[1] == self.dimension, "Query dimension mismatch"

        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.index.search(query, k)

        scores = scores[0].tolist()
        indices = indices[0].tolist()

        metadata_list = [
            self.metadata[i] for i in indices if i != -1
        ]

        return scores, metadata_list