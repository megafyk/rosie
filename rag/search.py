import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import EMBEDDING_MODEL_NAME
from rag.database import db


def semantic_search(query, embedding_model, k):
    embedding = np.array(embedding_model.embed_query(query))
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
    user_query = "Hello world"
    embedding_model_name = EMBEDDING_MODEL_NAME
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cpu"},
                                            encode_kwargs={"device": "cpu", "batch_size": 10})
    semantic_context = semantic_search(user_query, embedding_model, 10)
    context = "\n".join([c["text"] for c in semantic_context])

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(model_id)

    prompt = f"{user_query}\n\n{context}\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    output = llm.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
