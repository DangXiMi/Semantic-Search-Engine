import numpy as np
from sentence_transformers import SentenceTransformer
from src.indexer import FAISSIndex
from src.config import settings

# Load model (same as used for indexing)
model = SentenceTransformer(settings.MODEL_NAME)

# Load index and metadata
index = FAISSIndex()
index.load(settings.INDEX_FILE, settings.METADATA_FILE)

# Test query
query = "What is machine learning?"
query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

# Search
scores, results = index.search(query_vec, k=5)

# Display
print(f"Query: '{query}'\n")
for i, (score, res) in enumerate(zip(scores, results)):
    print(f"{i+1}. Score: {score:.4f}")
    print(f"   Title: {res['title']}")
    print(f"   Text: {res['text'][:150]}...\n")