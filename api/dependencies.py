# api/dependencies.py
from functools import lru_cache
from src.retriever import Retriever
from src.config import settings
from src.llm import GeminiProvider, OllamaProvider

@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """
    Singleton retriever loader.
    lru_cache ensures the model/index are loaded exactly once,
    even if this function is called multiple times.
    """
    return Retriever()


@lru_cache(maxsize=1)
def get_llm_provider(provider_name: str = None):
    """Return the appropriate LLM provider singleton."""
    if provider_name is None:
        provider_name = settings.LLM_PROVIDER  # defaults to "ollama" if not set in .env

    if provider_name == "gemini":
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env or environment")
        return GeminiProvider(api_key=api_key, model_name=settings.GEMINI_MODEL)
    else:
        return OllamaProvider()