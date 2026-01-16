# AeroSearch-Domain-Adaptive-Semantic-Search-for-Aerospace-Engineering-Documents
Designed and deployed a domain-adaptive NLP system for semantic search and recommendation over aerospace engineering documents using fine-tuned LLM embeddings.

# ✈️ AeroSearch  
## Domain-Adaptive Semantic Search for Aerospace Engineering Documents

AeroSearch is an NLP-based semantic search and recommendation system designed to improve engineering productivity by enabling intelligent retrieval of aerospace engineering documents. The system leverages Large Language Model (LLM) embeddings, domain-adaptive fine-tuning, and a web-based interface to simulate a real-world industrial AI tool.

---

## 🚀 Motivation

Aerospace engineering teams work with large volumes of technical documents such as inspection manuals, safety reports, and maintenance logs. Traditional keyword-based search systems often fail due to:

- Highly domain-specific vocabulary  
- Lack of labeled training data  
- Need for semantic understanding rather than exact matches  

AeroSearch addresses these challenges by using **semantic embeddings and domain-adaptive learning** to enable meaning-based document retrieval.

---

## 🧠 Key Features

- 🔍 **Semantic Search** using LLM embeddings (sentence-transformers)  
- 🧩 **Domain-Adaptive Fine-Tuning** on aerospace-specific text  
- 📊 **Evaluation using Precision@K** metrics  
- 🖥️ **Interactive Streamlit Web App** simulating an internal engineering tool  
- ⚙️ **Zero-Shot Retrieval** without labeled datasets  

---

## 🏗️ System Architecture

1. Aerospace documents are preprocessed and cleaned  
2. Documents are converted into vector embeddings using LLMs  
3. FAISS is used for efficient similarity-based retrieval  
4. User queries are embedded and matched semantically  
5. Top relevant documents are returned via a web interface  

---

## 🧪 Tech Stack

- **Programming Language:** Python  
- **NLP / LLMs:** Sentence Transformers  
- **Vector Search:** FAISS  
- **Web Framework:** Streamlit  
- **Evaluation:** Precision@K  

---



