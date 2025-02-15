import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import EMBEDDING_MODEL_NAME, TEST_PART_SYSTEM, TEST_PART_QUESTION, TEST_PART_CONTEXT, TEST_PART_ANSWER
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
    # Move model to GPU if available
    # device = 0 if torch.cuda.is_available() else -1
    device = "cpu"
    query = TEST_PART_QUESTION
    embedding_model_name = EMBEDDING_MODEL_NAME
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device},
                                            encode_kwargs={"device": device, "batch_size": 100})
    semantic_context = semantic_search(query, embedding_model, 10)
    context = "\n".join([c["text"] for c in semantic_context])

    prompt = f"""
{TEST_PART_SYSTEM}

{TEST_PART_QUESTION}

{TEST_PART_CONTEXT} {context}

{TEST_PART_ANSWER}
    """
    print(prompt)
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(model_id)

    max_retries = 3
    retry_count = 0
    retry_interval = 3
    temperature = 0.7
    max_new_tokens = 10

    while retry_count <= max_retries:
        try:

            # Create a text-generation pipeline.
            generator = pipeline(
                "text-generation",
                model=llm,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )

            # Generate a response.
            outputs = generator(prompt)
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)
            retry_count += 1
