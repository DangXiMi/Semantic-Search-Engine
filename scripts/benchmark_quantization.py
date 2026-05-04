# scripts/benchmark_quantization.py
import time, json, numpy as np
from pathlib import Path
from src.embedder_onnx import EmbedderONNX
from src.embedder import Embedder
from src.config import settings

def measure_throughput(embedder, texts, batch_size=32, num_runs=3):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = embedder.encode(texts, batch_size=batch_size, show_progress=False)
        times.append(time.perf_counter() - start)
    avg_time = sum(times) / len(times)
    return avg_time, len(texts) / avg_time

def main():
    # Load sample texts
    chunks_path = settings.DATA_DIR / "processed" / "chunks.jsonl"
    with open(chunks_path, 'r', encoding='utf-8') as f:
        texts = [json.loads(line)["text"] for line in f][:1000]

    print(f"Benchmarking CPU inference with {len(texts)} texts, batch_size=32")
    
    # FP32 ONNX embedder (force CPU)
    fp32_embedder = EmbedderONNX(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        onnx_path="models/minilm-onnx/model.onnx",
        providers=['CPUExecutionProvider']
    )
    int8_embedder = EmbedderONNX(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        onnx_path="models/minilm-int8/model_quantized.onnx",
        providers=['CPUExecutionProvider']
    )
    
    # Benchmark
    fp32_time, fp32_tp = measure_throughput(fp32_embedder, texts, batch_size=32)
    print(f"FP32 ONNX:  {fp32_time:.2f}s, {fp32_tp:.1f} texts/s")
    
    int8_time, int8_tp = measure_throughput(int8_embedder, texts, batch_size=32)
    print(f"INT8 ONNX:  {int8_time:.2f}s, {int8_tp:.1f} texts/s")
    
    speedup = int8_tp / fp32_tp
    print(f"Speedup (INT8 vs FP32): {speedup:.2f}x")
    
    # Quality check
    sample_texts = texts[:10]
    fp32_emb = fp32_embedder.encode(sample_texts, show_progress=False)
    int8_emb = int8_embedder.encode(sample_texts, show_progress=False)
    diff = np.abs(fp32_emb - int8_emb).max()
    print(f"Max difference (FP32 vs INT8): {diff:.6f}")

if __name__ == "__main__":
    main()