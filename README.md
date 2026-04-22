# Semantic Search Engine

Production-ready semantic search over Simple Wikipedia using sentence-transformers and FAISS.

## Features
- Embedding pipeline with `all-MiniLM-L6-v2`
- FAISS IndexFlatIP for exact cosine similarity search
- FastAPI REST API with OpenAPI docs
- Streamlit UI for interactive queries
- Evaluation suite with MRR and NDCG@5

## Architecture
[Insert a simple diagram or list components]

## Quick Start
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download sample data (link) or use your own
4. Run preprocessing: `python scripts/build_corpus.py`
5. Build embeddings: `python scripts/build_embeddings.py`
6. Build index: `python scripts/build_index.py`
7. Start API: `uvicorn api.app:app --reload`
8. Start UI: `streamlit run ui/streamlit_app.py`

## Evaluation Results
| Metric | Score |
|--------|-------|
| MRR    | 0.49  |
| NDCG@5 | 0.35  |

*See `evaluation/results.json` for per-query breakdown.*

## Project Structure
[Tree of important directories]

## Future Improvements
- IVF index for scaling
- Cross-encoder re-ranking
- Model versioning

## License
MIT