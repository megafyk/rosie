import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME, QUESTION, TEST_PART_SYSTEM, LLM_MODEL_NAME, OPENAI_API_KEY, OPENAI_MODEL
from database import db


def semantic_search(query, embedding_model, k):
    embedding = np.array(embedding_model.embed_query(query.lower().strip()))
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


def search(query, **kwargs):
    device = kwargs.get("device", "cpu")
    batch_size = kwargs.get("batch_size", 100)
    embedding_model_name = EMBEDDING_MODEL_NAME
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device},
                                            encode_kwargs={"device": device, "batch_size": batch_size})
    semantic_context = semantic_search(query, embedding_model, 10)
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
        {"role": "user", "content": query}
    ]

    from generate import generate
    # print(generate(f"local:{LLM_MODEL_NAME}", messages, {}))
    # print(generate(f"openai:{OPENAI_MODEL}", messages, {}))
    # print(generate(f"deepseek:deepseek-chat", messages, {}))
    # print(generate(f"google:gemini-2.0-flash", messages, {}))
