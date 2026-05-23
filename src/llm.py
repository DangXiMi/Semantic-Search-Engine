# src/llm.py (updated imports)
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict
from src.config import settings

from google import genai          
from google.genai import types    
import ollama

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> Dict:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = None):
        # Use settings.GEMINI_MODEL if no model_name passed
        self.model_name = model_name or settings.GEMINI_MODEL
        # NEW: Client-based, not global configure
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0,
                 max_retries: int = 3) -> Dict:
        for attempt in range(max_retries):
            try:
                start = time.time()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
                latency = (time.time() - start) * 1000
                usage = {}
                try:
                    metadata = response.usage_metadata
                    usage = {
                        "prompt_tokens": metadata.prompt_token_count,
                        "completion_tokens": metadata.candidates_token_count
                    }
                except Exception:
                    usage = {"prompt_tokens": 0, "completion_tokens": 0}
                return {"answer": response.text, "usage": usage, "latency_ms": latency}
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 15 * (2 ** attempt)
                    logger.warning(f"Rate limited. Retrying in {wait}s (attempt {attempt+1})...")
                    time.sleep(wait)
                else:
                    logger.error(f"Gemini generate failed: {e}")
                    raise
        raise RuntimeError("Max retries exceeded for Gemini API")

class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> Dict:
        start = time.time()
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )
            latency = (time.time() - start) * 1000
            # Ollama does not directly give token counts; estimate or leave as 0.
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            return {"answer": response["response"].strip(), "usage": usage, "latency_ms": latency}
        except Exception as e:
            logger.error(f"Ollama generate failed: {e}")
            raise