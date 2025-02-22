import json
from functools import partial
from timeit import default_timer as timer

import psycopg2
import ray
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pgvector.psycopg2 import register_vector

from config import EMBEDDING_MODEL_NAME, NUM_NODES, NUM_CPUS, NUM_GPUS, DB_NAME, DB_USER, DB_PORT, DB_HOST, DB_PASSWORD


class EmbedChunks:
    def __init__(self, model_name, device, batch_size=100):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"device": device, "batch_size": batch_size}
        )

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {
            "text": batch["text"],
            "source": batch["source"],
            "embeddings": embeddings
        }


class StoreEmbedding:
    def __call__(self, batch):
        with psycopg2.connect(
                f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} password={DB_PASSWORD}"
        ) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for text, source, embedding in zip(
                        batch["text"], batch["source"], batch["embeddings"]
                ):
                    cur.execute(
                        "INSERT INTO document (text, source, embedding) select %s, %s, %s WHERE NOT EXISTS (SELECT 1 FROM document WHERE text = %s)",
                        (
                            text,
                            source,
                            embedding,
                            text
                        ),
                    )
        return {}


def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.create_documents(
        texts=[f"{section['content']}"], metadatas=[{'source': section['page_id']}]
    )
    return [{'text': chunk.page_content, 'source': chunk.metadata['source']} for chunk in chunks]


def get_resources():
    num_cpus = int(ray.cluster_resources().get("CPU", 0)) if not NUM_CPUS else int(NUM_CPUS)
    num_gpus = int(ray.cluster_resources().get("GPU", 0)) if not NUM_GPUS else int(NUM_GPUS)
    num_nodes = len(ray.nodes()) if not NUM_NODES else int(NUM_NODES)
    print(f"Number of CPUs: {num_cpus}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Number of nodes: {num_nodes}")
    return {
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "num_nodes": num_nodes
    }


def build_index(data, chunk_size, chunk_overlap, batch_size, embedding_model_name):
    # docs -> sections -> chunks
    ds = ray.data.from_items(data)
    chunks_ds = ds.flat_map(
        partial(
            chunk_section,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    )
    resources = get_resources()

    # Embed chunks
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model_name,
                               "device": "cuda" if resources["num_gpus"] > 0 else "cpu"},
        batch_size=batch_size,
        num_cpus=resources["num_cpus"],
        num_gpus=resources["num_gpus"],
        concurrency=resources["num_nodes"],
    )

    # Index data
    embedded_chunks.map_batches(
        StoreEmbedding,
        batch_size=batch_size,
        num_cpus=resources["num_cpus"],
        num_gpus=resources["num_gpus"],
        concurrency=resources["num_nodes"]
    ).count()

    # Save to SQL dump
    # execute_bash(f"sudo -u postgres pg_dump -c > {sql_dump_fp}")

    print("Updated the index!")


if __name__ == "__main__":
    # read data from space_data.json line by line and chunk text into smaller parts
    print("----------------start load----------------")
    with open("../datasets/space_data.json") as file:
        space_data = json.load(file)
    print("----------------end load----------------")

    print("----------------start embedding----------------")

    embedding_model_name = EMBEDDING_MODEL_NAME

    chunk_size = 512
    chunk_overlap = 50
    batch_size = 100
    start_embeddings = timer()
    build_index(space_data, chunk_size, chunk_overlap, batch_size, embedding_model_name)
    end_embeddings = timer()

    print(f"\n----------------end embedding Elapsed {end_embeddings - start_embeddings}s----------------")
