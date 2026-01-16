from search_engine import SemanticSearchEngine
from preprocess import load_documents, clean_text

def precision_at_k(relevant_docs, retrieved_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_count = sum(
        1 for doc in retrieved_k if doc in relevant_docs
    )
    return relevant_count / k


if __name__ == "__main__":
    docs = load_documents("../data/documents.txt")
    docs = [clean_text(doc) for doc in docs]

    engine = SemanticSearchEngine(docs)

    query = "engine inspection crack"
    relevant_docs = [
        "jet engine blade inspection is critical for detecting cracks caused by fatigue."
    ]

    retrieved_docs = engine.search(query, top_k=5)

    p_at_5 = precision_at_k(relevant_docs, retrieved_docs, 5)
    print("Precision@5:", p_at_5)
