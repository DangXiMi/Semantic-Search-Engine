# scripts/evaluate.py
import json
from pathlib import Path
from typing import List, Dict, Set
from src.retriever import Retriever
from evaluation.metrics import precision_at_k, reciprocal_rank, ndcg_at_k
from src.config import settings

def load_golden_dataset(path: Path) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_query(retriever: Retriever, query_item: Dict, k: int = 5) -> Dict:
    query = query_item["query"]
    relevant_dict = {doc["id"]: doc["relevance"] for doc in query_item["relevant_docs"]}
    relevant_set = {doc["id"] for doc in query_item["relevant_docs"]}
    
    # Retrieve results
    results = retriever.search(query, k=k)
    retrieved_ids = [res["id"] for res in results]
    
    # Compute metrics
    p_at_k = precision_at_k(retrieved_ids, relevant_set, k)
    rr = reciprocal_rank(retrieved_ids, relevant_set)
    ndcg = ndcg_at_k(retrieved_ids, relevant_dict, k)
    
    return {
        "query": query,
        "precision": p_at_k,
        "reciprocal_rank": rr,
        f"ndcg@{k}": ndcg,
        "retrieved": retrieved_ids
    }

def main():
    dataset_path = Path("evaluation/golden_dataset.json")
    queries = load_golden_dataset(dataset_path)
    
    retriever = Retriever()
    
    total_precision = 0.0
    total_rr = 0.0
    total_ndcg = 0.0
    results = []
    
    for q in queries:
        metrics = evaluate_query(retriever, q, k=5)
        total_precision += metrics["precision"]
        total_rr += metrics["reciprocal_rank"]
        total_ndcg += metrics["ndcg@5"]
        results.append(metrics)
    
    n = len(queries)
    print(f"Evaluated {n} queries")
    print(f"Mean Precision@5: {total_precision/n:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {total_rr/n:.4f}")
    print(f"Mean NDCG@5: {total_ndcg/n:.4f}")
    
    # Save detailed results
    output_path = Path("evaluation/results.json")
    with open(output_path, 'w') as f:
        json.dump({"summary": {"mrr": total_rr/n, "ndcg@5": total_ndcg/n}, "details": results}, f, indent=2)
    print(f"Saved detailed results to {output_path}")

if __name__ == "__main__":
    main()