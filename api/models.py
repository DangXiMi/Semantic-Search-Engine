# api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    k: int = Field(5, ge=1, le=100, description="Number of results to return")

class SearchResultItem(BaseModel):
    id: str
    title: str
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total: int

class ErrorResponse(BaseModel):
    detail: str
    
class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="The user's question")
    k: int = Field(3, ge=1, le=10, description="Number of context chunks to include")
    use_llm: bool = True

class SourceDocument(BaseModel):
    id: str
    title: str
    snippet: str   # first 300 chars of the chunk
    
class AgentResponse(BaseModel):
    query: str
    answer: str = ""            # generated answer
    prompt: str = ""            # the constructed prompt (optional)
    sources: List[SourceDocument]
    suggested_queries: List[str] = []
    
class LLMProvider(str, Enum):
    gemini = "gemini"
    ollama = "ollama"

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The full prompt to send to the LLM")
    provider: LLMProvider = LLMProvider.gemini
    max_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.0, ge=0.0, le=2.0)

class GenerateResponse(BaseModel):
    answer: str
    model_used: str
    usage: Dict[str, int]   # e.g., {"prompt_tokens": 50, "completion_tokens": 30}
    latency_ms: float