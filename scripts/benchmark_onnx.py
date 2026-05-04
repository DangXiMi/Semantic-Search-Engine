# scripts/benchmark_onnx.py
import time
import json
import numpy as np
from pathlib import Path
from src.embedder import Embedder
from src.embedder_onnx import EmbedderONNX
from src.config import settings

def measure_throughput(embedder, texts, batch_size: int, num_runs: int = 3):
    """Measure encoding throughput in texts/second."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        embedder.encode(texts, batch_size=batch_size, show_progress=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    throughput = len(texts) / avg_time
    return avg_time, throughput

def main():
    # Load sample texts
    chunks_path = settings.DATA_DIR / "processed" / "chunks.jsonl"
    with open(chunks_path, 'r', encoding='utf-8') as f:
        texts = [json.loads(line)["text"] for line in f][:1000]  # First 1000 texts
    
    print(f"Benchmarking with {len(texts)} texts, batch_size=32")
    
    # PyTorch benchmark
    print("\n--- PyTorch Model ---")
    pytorch_embedder = Embedder()
    pytorch_time, pytorch_tp = measure_throughput(pytorch_embedder, texts, batch_size=32)
    print(f"Time: {pytorch_time:.2f}s | Throughput: {pytorch_tp:.1f} texts/s")
    
    # ONNX benchmark
    print("\n--- ONNX Model ---")
    onnx_embedder = EmbedderONNX()
    onnx_time, onnx_tp = measure_throughput(onnx_embedder, texts, batch_size=32)
    print(f"Time: {onnx_time:.2f}s | Throughput: {onnx_tp:.1f} texts/s")
    
    # Speedup
    speedup = onnx_tp / pytorch_tp
    print(f"\n--- Speedup: {speedup:.2f}x ---")
    
    # Also check output similarity
    print("\n--- Quality Check ---")
    sample_texts = texts[:10]
    pytorch_emb = pytorch_embedder.encode(sample_texts, show_progress=False)
    onnx_emb = onnx_embedder.encode(sample_texts, show_progress=False)
    diff = np.abs(pytorch_emb - onnx_emb).max()
    print(f"Max absolute difference: {diff:.6f}")
    if diff < 0.01:
        print("✓ Outputs are nearly identical")
    else:
        print("✗ Outputs differ significantly — check pooling/normalization")

if __name__ == "__main__":
    main()