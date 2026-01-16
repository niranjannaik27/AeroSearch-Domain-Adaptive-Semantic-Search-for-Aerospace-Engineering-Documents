def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        documents = f.readlines()
    return [doc.strip() for doc in documents if doc.strip()]


def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    return text


if __name__ == "__main__":
    docs = load_documents("../data/documents.txt")
    cleaned_docs = [clean_text(doc) for doc in docs]

    print("Sample cleaned document:")
    print(cleaned_docs[0])
