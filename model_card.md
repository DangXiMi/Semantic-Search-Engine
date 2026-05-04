# Model Card: Semantic Search Embedding

## Model Details
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Exported to ONNX:** Yes (FP32 and INT8 quantized)
- **Embedding dimension:** 384
- **Token limit:** 256 (truncation)

## Intended Use
- Semantic search over a corpus of Simple English Wikipedia articles (~37k chunks).
- Suitable for short-to-medium length queries (5-50 words).
- Designed for informational queries, not for highly domain-specific or real-time factual answers.

## Performance (on Golden Dataset)
| Metric | Original PyTorch | ONNX FP32* | ONNX INT8* |
|--------|------------------|------------|------------|
| MRR    | 0.49             | 0.49       | 0.52       |
| NDCG@5 | 0.35             | 0.35       | 0.36       |

*Values for ONNX are estimated to be identical to PyTorch (within 1e-4 difference). INT8 figures from one evaluation run.*

**Latency (CPU, batch=32 for 1000 texts):**
| Backend | Throughput (texts/s) |
|---------|-----------------------|
| PyTorch (GPU) | ~340 |
| ONNX FP32 (CPU) | 44.5 |
| ONNX INT8 (CPU) | 48.2 |

## Limitations & Known Weaknesses
- **Struggles with abstract or opinion-based queries:** e.g., "What were Abraham Lincoln's views on slavery?" returned general Lincoln content instead of the specific discussion.
- **Chunk fragmentation:** Long topics split across multiple chunks may lose context; a re-ranker could mitigate this.
- **Only English:** The model is trained on English data; queries in other languages will yield poor results.
- **Limited corpus:** Simple Wikipedia contains only basic knowledge; out-of-domain queries (e.g., quantum field theory) will retrieve unrelated documents.
- **Exact duplicates removed:** The deduplication step may drop chunks that are similar but not identical; some nuance could be lost.

## Evaluation
- Golden dataset of 42 manually annotated queries with graded relevance (0-3).
- Metrics: Precision@5, MRR, NDCG@5.
- Adversarial testing (to be expanded): short queries, ambiguous terms, out-of-domain terms.

## Trade-offs
- **Model size:** 80 MB (PyTorch) → 23 MB (ONNX INT8). Suitable for edge/CPU deployment.
- **Speed vs Quality:** INT8 gives a speedup on CPU with negligible accuracy loss.
- **Simplicity vs Re-ranking:** Using only a bi-encoder is fast but imperfect; adding a cross-encoder re-ranker would improve metrics but add latency.

## Ethical Considerations
- The corpus is from Wikipedia, which may contain biases. No personal or sensitive data is indexed.
- The search results are based on semantic similarity; they may not represent factual correctness.
- The system does not perform any content filtering beyond deduplication.

## Maintenance
- Retraining is not planned, but the embedding model and index can be swapped via versioned artifacts.
- FAISS index can be rebuilt with a different corpus by re-running the data pipeline.