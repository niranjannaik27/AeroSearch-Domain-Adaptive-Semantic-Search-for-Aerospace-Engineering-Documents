import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocess import load_documents, clean_text

class SemanticSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        # self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = SentenceTransformer("aerospace_finetuned_model")
        self.embeddings = self.model.encode(documents)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

    def search(self, query, top_k=5):
        query = clean_text(query)
        query_embedding = self.model.encode([query])

        distances, indices = self.index.search(
            np.array(query_embedding), top_k
        )

        results = []
        for idx in indices[0]:
            results.append(self.documents[idx])

        return results


if __name__ == "__main__":
    docs = load_documents("../data/documents.txt")
    docs = [clean_text(doc) for doc in docs]

    engine = SemanticSearchEngine(docs)

    query = "engine crack inspection"
    results = engine.search(query)

    print("\nSearch Results:")
    for r in results:
        print("-", r)
