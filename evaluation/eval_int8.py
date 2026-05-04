# scripts/eval_int8.py (based on your evaluate.py)
import json, numpy as np
from pathlib import Path
from src.embedder_onnx import EmbedderONNX
from src.indexer import FAISSIndex
from src.config import settings
from evaluation.metrics import precision_at_k, reciprocal_rank, ndcg_at_k

# Load golden dataset
dataset_path = Path("evaluation/golden_dataset.json")
queries = json.load(open(dataset_path, 'r', encoding='utf-8'))

# Load INT8 embedder and index
int8_embedder = EmbedderONNX(onnx_path="models/minilm-int8/model_quantized.onnx")
int8_embedder.session.set_providers(['CPUExecutionProvider'])  # CPU to avoid GPU overhead
index = FAISSIndex()
index.load(settings.INDEX_FILE, settings.METADATA_FILE)

k = 5
mrrs, ndcgs = [], []
for q in queries:
    query_text = q["query"]
    relevant_dict = {doc["id"]: doc["relevance"] for doc in q["relevant_docs"]}
    relevant_set = set(relevant_dict.keys())
    
    # Get embeddings (INT8)
    q_emb = int8_embedder.encode([query_text], show_progress=False).astype(np.float32)
    scores, metadata = index.search(q_emb, k=k)
    retrieved_ids = [m["id"] for m in metadata]
    
    rr = reciprocal_rank(retrieved_ids, relevant_set)
    ndcg_val = ndcg_at_k(retrieved_ids, relevant_dict, k)
    mrrs.append(rr)
    ndcgs.append(ndcg_val)

mrr = np.mean(mrrs)
ndcg = np.mean(ndcgs)
print(f"INT8 MRR: {mrr:.4f}")
print(f"INT8 NDCG@5: {ndcg:.4f}")