# scripts/build_embeddings.py
import json
from pathlib import Path
from src.embedder import Embedder
from src.config import settings

def main():
    chunks_file = settings.DATA_DIR / "processed" / "chunks.jsonl"
    output_file = settings.DATA_DIR / "embeddings" / "embeddings.npy"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    embedder = Embedder()
    embedder.encode_file(str(chunks_file), str(output_file), batch_size=64)
    
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()