import argparse
import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm


def read_txt_files(kb_dir: str) -> List[str]:
    docs = []
    for root, _, files in os.walk(kb_dir):
        for fn in files:
            if fn.endswith(".txt"):
                path = os.path.join(root, fn)
                with open(path, "r", encoding="utf-8") as f:
                    docs.append((fn, f.read()))
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--embedding", default="intfloat/multilingual-e5-base")
    args = parser.parse_args()

    docs = read_txt_files(args.kb_dir)

    client = chromadb.PersistentClient(path=args.out_dir)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.embedding
    )
    collection = client.get_or_create_collection(name="kb", embedding_function=embed_fn)

    ids, texts, metas = [], [], []
    for filename, content in tqdm(docs, desc="Chunking"):
        chunks = chunk_text(content, args.chunk_size, args.chunk_overlap)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{filename}-{idx}")
            texts.append(chunk)
            metas.append({"source": filename, "chunk_id": idx})

    collection.add(ids=ids, documents=texts, metadatas=metas)
    print(f"Indexed {len(ids)} chunks")


if __name__ == "__main__":
    main()
