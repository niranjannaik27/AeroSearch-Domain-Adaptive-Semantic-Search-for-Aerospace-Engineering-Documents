from sentence_transformers import SentenceTransformer
import numpy as np
from preprocess import load_documents, clean_text

def generate_embeddings(documents):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents)
    return embeddings


if __name__ == "__main__":
    docs = load_documents("../data/documents.txt")
    docs = [clean_text(doc) for doc in docs]

    embeddings = generate_embeddings(docs)

    print("Embedding shape:", embeddings.shape)
