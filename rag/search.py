from timeit import default_timer as timer

import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import EMBEDDING_MODEL_NAME, TEST_PART_SYSTEM, TEST_PART_QUESTION, TEST_PART_CONTEXT, TEST_PART_ANSWER
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

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id)


    pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500
    )

    llm = HuggingFacePipeline(pipeline=pipeline)


    template = """{test_part_system} {test_part_answer}
    {test_part_context} {context}
    {test_part_question}"""

    prompt_template = PromptTemplate(
        input_variables=["test_part_system", "test_part_answer", "test_part_context", "context", "test_part_question"],
        template=template
    )

    prompt = prompt_template.format(
        test_part_system=TEST_PART_SYSTEM,
        test_part_answer=TEST_PART_ANSWER,
        test_part_context=TEST_PART_CONTEXT,
        context=context,
        test_part_question=TEST_PART_QUESTION
    )

    print("Generated Prompt:")
    print(prompt)

    start_time = timer()
    outputs = llm(prompt)
    end_time = timer()
    print(outputs[0]["generated_text"])
    print(f"Elapsed {end_time - start_time}s")
