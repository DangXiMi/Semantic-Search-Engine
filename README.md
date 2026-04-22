# Semantic Search on Simple Wikipedia · End-to-End Retrieval System

**A production-minded semantic search engine over 36k Wikipedia chunks, built with transformers, FAISS, and FastAPI — complete with evaluation, streaming ingestion, and an interactive UI.**

---

## Overview

This project implements a full-stack semantic search system designed to surface relevant Wikipedia passages from natural language queries. It mirrors the core retrieval stack found in modern search and RAG applications: document ingestion, embedding generation, vector indexing, and a low‑latency API.

**Why it matters**: Semantic search is the foundation of retrieval-augmented generation, enterprise knowledge bases, and e‑commerce discovery. This project demonstrates the engineering discipline required to move from a model checkpoint to a reliable, measurable, and maintainable search service.

---

## Live Demo (Screenshots)

*Placeholder for UI screenshot showing query input and ranked results with relevance scores.*

*Placeholder for evaluation dashboard (MRR, NDCG, query‑level analysis).*

---

## Key Features

- **Streaming Ingestion** – Processes a 10k‑article JSONL corpus without loading the entire dataset into memory.
- **Semantic Chunking** – Splits articles into coherent, overlapping chunks with title‑context preservation.
- **Vector Search with FAISS** – Exact cosine similarity search over 36,869 normalized embeddings (IndexFlatIP).
- **REST API** – FastAPI endpoint serving sub‑150ms queries with dependency‑injected retriever.
- **Interactive UI** – Streamlit application with cached backend calls for a responsive experience.
- **Ranking Evaluation** – 42 hand‑labeled queries graded on a 3‑point scale; MRR and NDCG@5 measured.
- **Production‑Ready Practices** – SHA256 deduplication, lifecycle‑aware index loading, and strict separation of vectors and metadata.

---
## System Architecture

```markdown
            [SimpleWiki JSONL]
                    │
                    ▼
┌───────────────────────────────────────────────┐
│               Ingestion Layer                 │
│  • Streaming line‑by‑line parse               │
│  • Title‑aware chunking (overlap: 20 tokens)  │
│  • SHA256 deduplication                       │
└───────────────────┬───────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│            Embedding & Indexing               │
│  • Batch inference (all‑MiniLM‑L6‑v2, 384d)   │
│  • L2 normalization → cosine via inner product│
│  • FAISS IndexFlatIP (exact search)           │
│  • Vectors → .npy  |  Metadata → .pkl         │
└───────────────────┬───────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│               Serving Layer                   │
│  • FastAPI lifespan: load index once          │
│  • Dependency injection of retriever          │
│  • Endpoint: POST /search (100‑150ms p95)     │
└───────────────────┬───────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────────────┐
│  Streamlit UI │       │   Evaluation Suite    │
│  • @st.cache  │       │   • 42 labeled queries│
│  • Snippet    │       │   • MRR: 0.49         │
│    previews   │       │   • NDCG@5: 0.35      │
└───────────────┘       └───────────────────────┘

Flow summary: Streaming JSONL → chunking + dedup → batch embedding + normalization → FAISS index + metadata → FastAPI service → UI / evaluation.
```
---
## Tech Stack

**Machine Learning**  
`sentence-transformers` · `all-MiniLM-L6-v2` · `numpy`

**Vector Index**  
`FAISS` · IndexFlatIP · Cosine similarity via normalized embeddings

**Backend**  
`FastAPI` · `uvicorn` · Lifespan events · Dependency injection

**Frontend**  
`Streamlit` · `requests` · Session‑state caching

**Data Processing**  
Streaming JSONL · Custom chunking · `hashlib` (SHA256)

**Evaluation**  
Manual relevance judgments (3‑point scale) · MRR · NDCG@5

---

## How It Works

### 1. Ingestion & Preprocessing

- The Simple Wikipedia JSONL file is streamed line‑by‑line to keep memory usage constant.
- Each article is split into overlapping chunks (approx. 100–150 words) while preserving the article title as a prefix (e.g., *“Title: Machine Learning. Content: …”*).
- Duplicate chunks are identified via SHA256 hashing and discarded.

### 2. Embedding Generation

- Chunks are batched (typically 64–128) and passed through `all-MiniLM-L6-v2`, producing **384‑dimensional** dense vectors.
- Every vector is L2‑normalized, enabling cosine similarity search using FAISS’s efficient inner product index (`IndexFlatIP`).

### 3. Indexing

- Normalized vectors are stored in a `.npy` file; metadata (titles, chunk IDs, raw text) is saved as a `.pkl` file.
- The FAISS index is built from the vector matrix and saved alongside the metadata.

### 4. Serving

- FastAPI loads the index and metadata once during application startup (lifespan).
- The retriever is injected into the `/search` endpoint, keeping business logic decoupled and testable.
- Queries are embedded, normalized, and searched with `index.search()` (k results, default k=10).

### 5. User Interface

- Streamlit provides a simple search bar. Results are displayed with titles, relevance scores, and snippet previews.
- `@st.cache_resource` caches the embedding model; `@st.cache_data` caches recent query results to reduce backend pressure.

### 6. Evaluation

- A fixed set of 42 queries was manually annotated with graded relevance (3 = highly relevant, 2 = partially relevant, 1 = loosely related).
- **MRR (Mean Reciprocal Rank): 0.49** — indicates reasonable performance on top‑1 retrieval.
- **NDCG@5: 0.35** — reflects decent ranking quality, with room for improvement on ambiguous queries.

---

## Performance & Evaluation

| Metric       | Value    |
|--------------|----------|
| Corpus size  | 36,869 chunks |
| Index size   | ~56 MB (in‑memory) |
| Embedding dim| 384     |
| Avg. API latency | 100–150 ms |
| MRR          | 0.49    |
| NDCG@5       | 0.35    |

### Interpretation

- **MRR of 0.49** means the first relevant result appears, on average, around rank 2. For a zero‑shot model without fine‑tuning, this is a solid baseline.
- **NDCG@5 of 0.35** shows that the system often places highly relevant documents near the top, but struggles when queries are underspecified or contain broad terms (e.g., “history”).
- **Failure cases** often involve polysemy (e.g., “bass” as fish vs. music) and temporal queries (e.g., “current president”) — both addressable with a cross‑encoder reranker.

---

## Engineering Decisions

### Why `all-MiniLM-L6-v2`?
- **Tradeoff**: 384‑dim vectors offer an excellent balance between embedding quality and memory footprint. The model is small enough to run on CPU with sub‑100ms inference, yet powerful enough for semantic matching on general‑domain text.

### Why FAISS IndexFlatIP?
- **Decision**: Exact search is feasible for 37k vectors (56 MB) and eliminates approximation noise. Cosine similarity via inner product avoids the runtime cost of explicit normalization per query.  
- **Production Consideration**: For 10x larger corpora, switching to IVF or HNSW would be trivial with the current modular retriever.

### Streaming + Deduplication
- **Rationale**: Memory‑constrained environments (e.g., cloud functions) benefit from streaming. SHA256 deduplication prevents nearly‑identical chunks (e.g., boilerplate disclaimers) from polluting the index and skewing results.

### Title Prefixing
- **Impact**: Chunks often lose context when isolated. Prepending the article title adds a lightweight form of “document‑aware” embedding without complex hierarchical models.

### Separation of Vectors and Metadata
- **Maintainability**: Enables vector index updates (e.g., re‑indexing with a new model) without touching metadata, and vice‑versa. Essential for model versioning in production.

### FastAPI Lifespan + Dependency Injection
- **Production Pattern**: Loading the index once at startup avoids per‑request I/O. Dependency injection makes the retriever swappable for A/B testing and unit testing.

### Streamlit Caching
- **User Experience**: Caching the embedding model and recent search results prevents repeated expensive computations, keeping the UI snappy even on modest hardware.

---

## Running Locally

**Prerequisites**: Python 3.9+, `pip`

```bash
# 1. Clone and set up environment
git clone https://github.com/yourusername/semantic-search-wikipedia.git
cd semantic-search-wikipedia
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Download the Simple Wikipedia JSONL (if not included)
#    Place it at data/simplewiki.jsonl

# 3. Run the ingestion & indexing pipeline
python scripts/build_index.py --input data/simplewiki.jsonl --output_dir index/

# 4. Start the FastAPI server
uvicorn app.main:app --reload --port 8000

# 5. Launch Streamlit UI (in a new terminal)
streamlit run ui/app.py
```

---
## Project Structure
``` markdown
.
├── app/
│   ├── main.py               # FastAPI app, lifespan, /search endpoint
│   ├── retriever.py          # Encapsulated FAISS search logic
│   └── models.py             # Pydantic request/response schemas
├── ui/
│   └── app.py                # Streamlit UI with caching
├── scripts/
│   ├── build_index.py        # End-to-end indexing pipeline
│   ├── chunker.py            # Streaming chunking & dedup
│   └── embed.py              # Batch embedding & normalization
├── evaluation/
│   ├── queries.json          # 42 labeled queries + relevance
│   └── evaluate.py           # MRR, NDCG calculation
├── index/
│   ├── vectors.npy           # Normalized 384‑dim vectors
│   ├── metadata.pkl          # Titles, chunk IDs, raw text
│   └── faiss.index           # FAISS IndexFlatIP
├── data/                     # Raw JSONL (gitignored)
├── requirements.txt
└── README.md
```
---
## Future Improvements


| Area | Concrete Next Step | Expected Impact |
|------|-------------------|-----------------|
| **Index Scalability** | Migrate to `IndexIVFFlat` with 256 centroids, add product quantization (PQ) for >1M vectors. | Maintain <50ms latency with 10x corpus growth; index size reduced ~75%. |
| **Ranking Precision** | Add cross‑encoder re‑ranking (`ms-marco-MiniLM-L-6-v2`) over top‑20 candidates. | NDCG@5 projected to reach 0.50–0.55; polysemy and broad‑query failures mitigated. |
| **Query Caching** | Implement TTL‑aware LRU cache at FastAPI layer (e.g., `cachetools` with 1000 entry limit). | Eliminate redundant embedding and search for identical queries; 2–3x throughput improvement on repeated traffic. |
| **Model Versioning** | Extend metadata schema to include `model_name` and `embedding_dim`; support runtime selection via `/search?version=v2`. | Enables zero‑downtime A/B testing and rollback. |
| **Observability** | Add OpenTelemetry instrumentation for latency percentiles and index memory usage; expose `/metrics` for Prometheus. | Production readiness — alerts on p99 degradation or memory pressure. |
| **Deployment** | Provide `Dockerfile` with multi‑stage build and `docker-compose.yml` for local stack (API + UI). | One‑command reproducibility; foundation for cloud deployment (AWS/GCP). |

---

## Why This Project Stands Out

This is **not** a "toy" semantic search demo. It's built with the same patterns used in production retrieval systems:

- **Streaming ingestion** and **deduplication** demonstrate data-engineering awareness suitable for large-scale, memory-constrained environments.
- **Exact evaluation with graded relevance** (42 hand-labeled queries, MRR 0.49, NDCG@5 0.35) shows commitment to measurable quality, not anecdotal "it works."
- **Clean separation of concerns** (ingestion, indexing, serving, evaluation) makes the codebase extensible and testable—a prerequisite for collaborative development.
- **Performance is quantified**: API latency and ranking metrics are upfront and interpreted, indicating a mindset oriented toward production SLAs.

For a recruiter or hiring manager, this project signals an engineer who understands the **full lifecycle** of an ML-powered feature, from data pipeline to user-facing API and offline evaluation.