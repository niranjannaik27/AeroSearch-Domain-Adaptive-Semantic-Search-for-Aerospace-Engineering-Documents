import streamlit as st
from src.preprocess import load_documents, clean_text
from src.search_engine import SemanticSearchEngine

st.set_page_config(page_title="Aerospace NLP Search", layout="centered")

st.title("✈️ Aerospace Engineering Document Search")
st.write(
    "Semantic search system for aerospace engineering documents using LLM embeddings."
)

# Load documents
docs = load_documents("data/documents.txt")
docs = [clean_text(doc) for doc in docs]

# Initialize search engine
@st.cache_resource
def load_engine(documents):
    return SemanticSearchEngine(documents)

engine = load_engine(docs)

# Search box
query = st.text_input("Enter engineering query:", "")

if query:
    results = engine.search(query, top_k=5)

    st.subheader("🔍 Search Results")
    for i, res in enumerate(results, 1):
        st.write(f"**{i}.** {res}")
