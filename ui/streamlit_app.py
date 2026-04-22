import streamlit as st
import requests
from typing import List, Dict

API_URL = "http://localhost:8000/search"

st.set_page_config(page_title="Semantic Search", layout="wide")
st.title("🔍 Semantic Search Engine")
st.markdown("Search Simple Wikipedia using AI embeddings.")

@st.cache_resource
def get_session() -> requests.Session:
    return requests.Session()

def search(query: str, k: int) -> List[Dict]:
    session = get_session()
    try:
        response = session.post(API_URL, json={"query": query, "k": k}, timeout=10)
        response.raise_for_status()
        return response.json()["results"]
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return []

# UI Widgets
query = st.text_input("Enter your search query:", placeholder="e.g., machine learning")
k = st.slider("Number of results:", min_value=1, max_value=20, value=5)

if st.button("Search") and query:
    with st.spinner("Searching..."):
        results = search(query, k)
    
    if results:
        st.success(f"Found {len(results)} results")
        for i, res in enumerate(results):
            with st.container():
                st.markdown(f"### {i+1}. {res['title']}  (Score: {res['score']:.3f})")
                st.markdown(res['text'][:500] + "..." if len(res['text']) > 500 else res['text'])
                st.divider()
    else:
        st.warning("No results found.")