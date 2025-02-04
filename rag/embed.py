import json
from functools import partial

import ray
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from ray.data import ActorPoolStrategy


def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.create_documents(
        texts=[f"{section['title']} {section['content']}"], metadatas=[{'source': section['page_id']}]
    )
    return [{'text': chunk.page_content, 'source': chunk.metadata['source']} for chunk in chunks]


class EmbedChunks:
    def __init__(self, model_name):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100})

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings":
            embeddings}


if __name__ == "__main__":
    # read data from space_data.json line by line and chunk text into smaller parts
    print("----------------start chunk----------------")

    with open("../datasets/space_data.json") as file:
        space_data = json.load(file)
    chunk_size = 300
    chunk_overlap = 50
    cnt = 0
    ds = ray.data.from_items(space_data)
    chunks_ds = ds.flat_map(partial(chunk_section, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    chunks_ds.show()
    print(f"\n{chunks_ds.count()} chunks")

    print("----------------end chunk----------------")

    print("----------------start embed----------------")
    # embed text chunks
    batch_size = 100
    num_gpus = 1
    num_compute = 6
    embedding_model_name = "dangvantuan/vietnamese-embedding"
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model_name},
        batch_size=batch_size,
        num_gpus=num_gpus,
        compute=ActorPoolStrategy(size=num_compute))
    embedded_chunks.show()
    print(f"{embedded_chunks.count()} chunks")
    print("----------------end embed----------------")
