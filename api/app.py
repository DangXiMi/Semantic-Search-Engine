# api/app.py
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from typing import Dict, Any
import logging

from api.models import SearchRequest, SearchResponse, ErrorResponse
from api.dependencies import get_retriever
from src.retriever import Retriever

from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler: runs at startup and shutdown.
    This is the correct place to load heavy resources.
    """
    # Startup: load retriever (model + index)
    logger.info("Loading embedding model and FAISS index...")
    retriever = get_retriever()  # Triggers the cache load
    app.state.retriever = retriever
    logger.info("Ready to serve requests.")
    
    yield  # Server is running here
    
    # Shutdown: cleanup (if needed)
    logger.info("Shutting down...")
    # FAISS index and model will be garbage collected

# Create FastAPI app with lifespan
app = FastAPI(
    title="Semantic Search API",
    description="Search Simple Wikipedia using sentence embeddings and FAISS.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],   # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

@app.post("/search", response_model=SearchResponse, responses={400: {"model": ErrorResponse}})
async def search(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever)
) -> SearchResponse:
    """
    Perform semantic search over the indexed Wikipedia corpus.
    """
    try:
        logger.info(f"Search query: '{request.query}' (k={request.k})")
        results = retriever.search(request.query, k=request.k)
        
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results)
        )
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")