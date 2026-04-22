from src.retriever import Retriever
import json

r = Retriever()
results = r.search("What is the capital of France?", k=3)
for res in results:
    print(f"{res['title']} (score: {res['score']:.3f})")
    
print(json.dumps(results[0], indent=2))