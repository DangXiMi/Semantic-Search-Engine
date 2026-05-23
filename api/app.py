# api/app.py
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from typing import Dict, Any
import logging

from api.models import SearchRequest, SearchResponse, ErrorResponse
from api.dependencies import get_retriever
from src.retriever import Retriever

from fastapi.middleware.cors import CORSMiddleware

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import get_current_span

# Phoenix OTLP endpoint – adjust for Docker vs local
PHOENIX_ENDPOINT = "http://127.0.0.1:6006/v1/traces"

trace_provider = TracerProvider()
trace_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT))
)
trace.set_tracer_provider(trace_provider)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler: runs at startup and shutdown.
    This is the correct place to load heavy resources.
    """
    logger.info("Loading embedding model and FAISS index...")
    retriever = get_retriever()  
    app.state.retriever = retriever
    logger.info("Ready to serve requests.")
    
    yield  
    
    logger.info("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Semantic Search API",
    description="Search Simple Wikipedia using sentence embeddings and FAISS.",
    version="1.0.0",
    lifespan=lifespan,
)

FastAPIInstrumentor.instrument_app(app)

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
    span = get_current_span()
    span.set_attribute("search.query", request.query)
    span.set_attribute("search.k", request.k)
    
    try:
        results = retriever.search(request.query, k=request.k)
        span.set_attribute("search.num_results", len(results))
        if results:
            span.set_attribute("search.top_score", results[0]["score"])
            span.set_attribute("search.top_title", results[0]["title"])
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results)
        )
    except Exception as e:
        span.set_attribute("error", True)
        span.set_attribute("error.message", str(e))
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")