# api/app.py
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from typing import Dict, Any
import logging

from api.models import SearchRequest, SearchResponse, ErrorResponse,AgentRequest, AgentResponse, SourceDocument,GenerateRequest, GenerateResponse, LLMProvider as LLMProviderEnum
from api.dependencies import get_retriever, get_llm_provider
from src.retriever import Retriever

from fastapi.middleware.cors import CORSMiddleware

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import get_current_span

from src.llm import LLMProvider as LLMProviderClass

# Phoenix OTLP endpoint – adjust for Docker vs local
# If running app locally: "http://127.0.0.1:6006/v1/traces"
# If running inside Docker: "http://host.docker.internal:6006/v1/traces"
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
    # Startup: load retriever (model + index)
    logger.info("Loading embedding model and FAISS index...")
    retriever = get_retriever()  # Triggers the cache load
    app.state.retriever = retriever
    logger.info("Ready to serve requests.")
    
    yield  
    
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
    retriever: Retriever = Depends(get_retriever) ) -> SearchResponse:
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
        return SearchResponse(query=request.query, results=results, total=len(results))
    except Exception as e:
        span.set_attribute("error", True)
        span.set_attribute("error.message", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    provider = get_llm_provider(request.provider.value)
    result = provider.generate(request.prompt, max_tokens=request.max_tokens, temperature=request.temperature)
    return GenerateResponse(
        answer=result["answer"],
        model_used=request.provider.value,
        usage=result["usage"],
        latency_ms=result["latency_ms"]
    )

@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(
    request: AgentRequest,
    retriever: Retriever = Depends(get_retriever),
    provider: LLMProviderClass = Depends(get_llm_provider)   # added LLM provider dependency
):
    span = get_current_span()
    span.set_attribute("agent.query", request.query)
    span.set_attribute("agent.k", request.k)
    span.set_attribute("agent.use_llm", request.use_llm)

    # Retrieve context (unchanged)
    results = retriever.search(request.query, k=request.k)
    contexts = []
    sources = []
    for i, res in enumerate(results):
        context_entry = f"[{i+1}] {res['title']}: {res['text'][:500]}"
        contexts.append(context_entry)
        sources.append(SourceDocument(
            id=res["id"],
            title=res["title"],
            snippet=res["text"][:300]
        ))
    context_block = "\n\n".join(contexts)

    system_prompt = "You are a helpful research assistant. Use the provided context to answer the user's question. If the answer cannot be found, say so."
    user_prompt = f"""Context:
{context_block}

Question: {request.query}

Answer:"""

    answer = None
    if request.use_llm:
        try:
            result = provider.generate(system_prompt + user_prompt, max_tokens=256, temperature=0.0)
            answer = result["answer"]
            span.set_attribute("agent.llm.latency_ms", result["latency_ms"])
            span.set_attribute("agent.llm.provider", provider.__class__.__name__)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            answer = f"[LLM Error] {str(e)}"

    if not answer:
        answer = f"[LLM not available] Prompt: {user_prompt[:200]}..."

    # Suggested queries (keep as is for now)
    words = request.query.split()
    suggested = [
        f"Tell me more about {request.query}",
        f"Explain {request.query} in simple terms",
        f"What are the implications of {request.query}?"
    ]

    return AgentResponse(
        query=request.query,
        answer=answer,
        prompt=user_prompt,
        sources=sources,
        suggested_queries=suggested
    )