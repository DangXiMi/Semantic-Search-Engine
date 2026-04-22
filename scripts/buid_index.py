# scripts/build_index.py
import json
import numpy as np
from pathlib import Path
from src.indexer import FAISSIndex
from src.config import settings

def main():
    # Paths
    embeddings_path = settings.DATA_DIR / "embeddings" / "embeddings.npy"
    chunks_path = settings.DATA_DIR / "processed" / "chunks.jsonl"
    index_path = settings.INDEX_FILE
    metadata_path = settings.METADATA_FILE
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Load embeddings and metadata
    embeddings = np.load(embeddings_path).astype(np.float32)
    with open(chunks_path, 'r', encoding='utf-8') as f:
        metadata = [json.loads(line) for line in f]

    # Verify alignment
    assert len(embeddings) == len(metadata), "Mismatch between embeddings and chunks"

    # Build and save
    index = FAISSIndex(dimension=embeddings.shape[1])
    index.build(embeddings, metadata)
    index.save(index_path, metadata_path)

    print(f"Index saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Total vectors indexed: {index.index.ntotal}")

if __name__ == "__main__":
    main()