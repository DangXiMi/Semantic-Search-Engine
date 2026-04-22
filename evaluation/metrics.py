# evaluation/metrics.py
import numpy as np
from typing import List, Dict, Set

def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = [doc_id for doc_id in retrieved_k if doc_id in relevant_ids]
    return len(relevant_retrieved) / k

def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Reciprocal of the rank of the first relevant document (1-indexed)."""
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0

def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    relevance_scores = relevance_scores[:k]
    if not relevance_scores:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
    return np.sum(relevance_scores / discounts)

def ndcg_at_k(retrieved_ids: List[str], relevance_dict: Dict[str, int], k: int) -> float:
    """
    Normalized DCG at k.
    relevance_dict maps doc_id -> relevance grade (0-3).
    """
    # TODO: Get relevance scores in order of retrieved_ids
    retrieved_relevances = [relevance_dict.get(doc_id, 0) for doc_id in retrieved_ids]
    
    # Compute DCG
    dcg = dcg_at_k(retrieved_relevances, k)
    
    # Compute IDCG (ideal DCG) by sorting relevance scores descending
    ideal_relevances = sorted(relevance_dict.values(), reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg