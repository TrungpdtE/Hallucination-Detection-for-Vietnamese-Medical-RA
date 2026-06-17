import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm


def tokenize(text: str) -> List[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


def read_txt_files(kb_dir: str) -> List[Tuple[str, str]]:
    docs = []
    for root, _, files in os.walk(kb_dir):
        for filename in files:
            if filename.endswith(".txt"):
                path = os.path.join(root, filename)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    docs.append((filename, f.read()))
    return docs


class BM25Index:
    def __init__(self, docs: List[Tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_terms = [Counter(tokenize(text)) for _, text in docs]
        self.doc_lens = [sum(counter.values()) for counter in self.doc_terms]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.df = defaultdict(int)
        for counter in self.doc_terms:
            for term in counter:
                self.df[term] += 1

    def score(self, query: str) -> List[Tuple[str, float]]:
        query_terms = tokenize(query)
        n_docs = len(self.docs)
        scores = []
        for idx, counter in enumerate(self.doc_terms):
            score = 0.0
            doc_len = self.doc_lens[idx] or 1
            for term in query_terms:
                tf = counter.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1 + (n_docs - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += idf * (tf * (self.k1 + 1) / denom)
            scores.append((self.docs[idx][0], score))
        return sorted(scores, key=lambda item: item[1], reverse=True)


def minmax(scores: List[Tuple[str, float]]) -> Dict[str, float]:
    if not scores:
        return {}
    values = [score for _, score in scores]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {doc_id: 0.0 for doc_id, _ in scores}
    return {doc_id: (score - lo) / (hi - lo) for doc_id, score in scores}


def dense_scores(vectorstore: str, embedding: str, query: str, top_k: int) -> List[Tuple[str, float]]:
    client = chromadb.PersistentClient(path=vectorstore)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding)
    collection = client.get_collection(name="medical_kb", embedding_function=embed_fn)
    result = collection.query(query_texts=[query], n_results=top_k)
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [(doc_id.split("-")[0], 1.0 / (1.0 + distance)) for doc_id, distance in zip(ids, distances)]


def hybrid_search(
    query: str,
    bm25: BM25Index,
    vectorstore: str,
    embedding: str,
    top_k: int,
    alpha: float,
) -> List[Dict]:
    bm25_raw = bm25.score(query)[: max(top_k * 10, 50)]
    dense_raw = dense_scores(vectorstore, embedding, query, max(top_k * 5, 25))
    bm25_norm = minmax(bm25_raw)
    dense_norm = minmax(dense_raw)
    doc_ids = set(bm25_norm) | set(dense_norm)
    ranked = []
    for doc_id in doc_ids:
        score = alpha * dense_norm.get(doc_id, 0.0) + (1 - alpha) * bm25_norm.get(doc_id, 0.0)
        ranked.append(
            {
                "doc_id": doc_id,
                "hybrid_score": round(score, 6),
                "dense_score": round(dense_norm.get(doc_id, 0.0), 6),
                "bm25_score": round(bm25_norm.get(doc_id, 0.0), 6),
            }
        )
    return sorted(ranked, key=lambda item: item["hybrid_score"], reverse=True)[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_dir", required=True)
    parser.add_argument("--vectorstore", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--embedding", default="intfloat/multilingual-e5-base")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55, help="Dense weight; BM25 weight is 1-alpha")
    args = parser.parse_args()

    docs = read_txt_files(args.kb_dir)
    for _ in tqdm(docs, desc="Indexing BM25"):
        pass
    bm25 = BM25Index(docs)
    results = hybrid_search(args.question, bm25, args.vectorstore, args.embedding, args.top_k, args.alpha)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
