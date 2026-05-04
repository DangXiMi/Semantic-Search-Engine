# scripts/setup.py
"""
One-command setup for new users:
Downloads the SimpleWiki dataset and builds the index.
Usage: python scripts/setup.py
"""
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # Step 0: Check if data already exists
    raw = Path("data/raw/simplewiki_10k.jsonl")
    if not raw.exists():
        print("Downloading SimpleWiki dataset...")
        run("python data/download_wiki.py")  
    else:
        print("Dataset already present.")

    # Step 1: Build corpus (chunks)
    if not Path("data/processed/chunks.jsonl").exists():
        run("python scripts/build_corpus.py")
    else:
        print("Chunks already built.")

    # Step 2: Build embeddings
    if not Path("data/embeddings/embeddings.npy").exists():
        run("python scripts/build_embeddings.py")
    else:
        print("Embeddings already built.")

    # Step 3: Build FAISS index
    if not Path("data/index/index.faiss").exists():
        run("python scripts/build_index.py")
    else:
        print("Index already built.")

    print("\n Setup complete. You can now run: docker compose up")
    print("Then open http://localhost:8000/docs")

if __name__ == "__main__":
    main()