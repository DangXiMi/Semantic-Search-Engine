# scripts/benchmark_latency.py
import time
import json
import requests
import numpy as np
from pathlib import Path
from typing import List

# ============================================================
# CONFIGURATION
# ============================================================
API_URL = "http://localhost:8000/search"
NUM_WARMUP = 5          # warmup requests to remove cold‑start effects
NUM_RUNS = 50           # number of measurements per query
QUERIES = [
    "machine learning",
    "capital of France",
    "How was Abraham Lincoln educated?",
    "What were Abraham Lincoln's views on slavery?",
    "Apollo 11 mission importance",
    "Albert Einstein discoveries",
    "What is Apple Inc. known for?",
    "ancient Egypt daily life",
    "Where is Afghanistan?",
    "alkali metals properties",
]
# ============================================================

def send_request(session: requests.Session, query: str, k: int = 5) -> float:
    """Send a search request and return response time in seconds."""
    start = time.perf_counter()
    resp = session.post(API_URL, json={"query": query, "k": k}, timeout=10)
    resp.raise_for_status()
    elapsed = time.perf_counter() - start
    return elapsed

def benchmark_queries(queries: List[str], num_runs: int) -> dict:
    """Run multiple requests and return latency percentiles."""
    session = requests.Session()
    all_times = []

    # Warmup
    print(f"Warming up ({NUM_WARMUP} requests)...")
    for _ in range(NUM_WARMUP):
        send_request(session, "warmup", k=1)

    # Measurement
    print(f"Benchmarking {len(queries)} queries × {num_runs} runs...")
    for q in queries:
        for _ in range(num_runs):
            try:
                t = send_request(session, q)
                all_times.append(t)
            except Exception as e:
                print(f"Request failed for query '{q}': {e}")

    times = np.array(all_times)
    stats = {
        "num_requests": len(times),
        "mean": float(np.mean(times)),
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }
    return stats

def main():
    print("API Latency Benchmark")
    print("Make sure the API is running (docker compose up or uvicorn).\n")
    
    stats = benchmark_queries(QUERIES, NUM_RUNS)
    
    print("\n--- Results ---")
    print(f"Total requests: {stats['num_requests']}")
    print(f"Mean latency:  {stats['mean']*1000:.1f} ms")
    print(f"p50 latency:   {stats['p50']*1000:.1f} ms")
    print(f"p95 latency:   {stats['p95']*1000:.1f} ms")
    print(f"p99 latency:   {stats['p99']*1000:.1f} ms")
    print(f"Min latency:   {stats['min']*1000:.1f} ms")
    print(f"Max latency:   {stats['max']*1000:.1f} ms")

    # Save to JSON for README
    output_path = Path("evaluation/latency_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved detailed results to {output_path}")

if __name__ == "__main__":
    main()