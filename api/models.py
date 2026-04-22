# api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

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