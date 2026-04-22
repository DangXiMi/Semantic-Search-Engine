# api/dependencies.py
from functools import lru_cache
from src.retriever import Retriever

@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """
    Singleton retriever loader.
    lru_cache ensures the model/index are loaded exactly once,
    even if this function is called multiple times.
    """
    return Retriever()