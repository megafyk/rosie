import numpy as np

from rag.config import EMBEDDING_MODEL_NAME
from rag.database import db
from langchain_huggingface import HuggingFaceEmbeddings


def semantic_search(query, embedding_model, k):
    embedding = np.array(embedding_model.embed_query(query))
    print(embedding)
    query = "SELECT * FROM document ORDER BY embedding <=> %s LIMIT %s"
    rows = db.execute_query(query, (embedding, k))
    semantic_context = [{"id": row[0], "text": row[1], "source": row[2]} for row in rows]
    return semantic_context


def lexical_search(index, query, chunks, k):
    query_tokens = query.lower().split()  # preprocess query
    scores = index.get_scores(query_tokens)  # get best matching (BM) scores
    indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]  # sort and get top k
    lexical_context = [
        {"id": chunks[i][0], "text": chunks[i][1], "source": chunks[i][2], "score": scores[i]}
        for i in indices
    ]
    return lexical_context


if __name__ == "__main__":
    embedding_model_name = EMBEDDING_MODEL_NAME
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cuda"}, encode_kwargs={"device": "cuda", "batch_size": 100})
    res = semantic_search("hello world", embedding_model, 100)
    print(res)