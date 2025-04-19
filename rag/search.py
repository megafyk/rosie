import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, OPENROUTER_MODEL, QUESTION, TEST_PART_SYSTEM
from generate import generate, generate_openrouter
from dotenv import load_dotenv

load_dotenv()

script_dir = Path(__file__).parent
faiss_index_path = script_dir.parent / "datasets" / "faiss_index.bin"
faiss_document_path = script_dir.parent / "datasets" / "faiss_documents.pkl"
# Load FAISS index
index = faiss.read_index(str(faiss_index_path))
with open(faiss_document_path, "rb") as f:
    documents = pickle.load(f)


def semantic_search(query, embedding_model, k=10):
    # Create query embedding
    query_vector = np.array([embedding_model.embed_query(query.lower().strip())]).astype('float32')

    # Search the index
    distances, indices = index.search(query_vector, k)

    # Get the corresponding documents
    results = []
    for i, idx in enumerate(indices[0]):
        if len(documents) > idx >= 0:  # Check if index is valid
            doc = documents[idx]
            results.append({
                "id": idx,
                "text": doc["text"],
                "source": doc["source"],
                "distance": float(distances[0][i])
            })

    return results


def search(query, **kwargs):
    device = kwargs.get("device", "cpu")
    batch_size = kwargs.get("batch_size", 100)
    embedding_model_name = EMBEDDING_MODEL_NAME
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": batch_size},
    )
    semantic_context = semantic_search(query, embedding_model, kwargs.get("k", 10))
    context = "\n\n".join([c["text"] for c in semantic_context])
    return context


if __name__ == "__main__":
    # Move model to GPU if available
    # device = 0 if torch.cuda.is_available() else -1

    query = QUESTION
    context = search(query)
    messages = [
        {"role": "system", "content": TEST_PART_SYSTEM},
        {"role": "assistant", "content": context},
        {"role": "user", "content": query},
    ]
    print(messages)
    res = generate_openrouter(
        os.getenv("OPENROUTER_MODEL"),
        messages,
        {
            "temperature": 0.0,
        }
    )
    print(res.choices[0].message.content)
