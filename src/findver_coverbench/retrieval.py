"""Hybrid retrieval: BM25 and dense (Qwen3-Embedding-4B) rankings merged with
reciprocal rank fusion. The same retriever serves RAG and agentic RAG.

Each document (a FinDVer report, or a CoverBench claim's source document) has
a pickled index with ``chunk_ids``, ``texts`` and ``embeddings``
(see build_index.py). Query embeddings come from the OpenRouter embeddings API.
"""

import json
import os
import pickle
import re
import time
from math import log2

import numpy as np
import requests

EMBED_MODEL = "qwen/qwen3-embedding-4b"
RRF_K = 60


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def embed(texts, max_retries=5, backoff=5):
    """Embeddings for a list of texts (OpenRouter)."""
    headers = {"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
               "Content-Type": "application/json"}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post("https://openrouter.ai/api/v1/embeddings", headers=headers,
                                 json={"model": EMBED_MODEL, "input": texts}, timeout=120)
            body = resp.json()
        except Exception as e:
            print(f"[embed] attempt {attempt}/{max_retries} failed: {e!r}")
            time.sleep(backoff * attempt)
            continue
        if resp.status_code == 200 and "data" in body:
            return np.array([d["embedding"] for d in body["data"]], dtype=np.float32)
        if resp.status_code in (401, 402, 403, 404):
            raise RuntimeError(f"embeddings call failed with status {resp.status_code}: "
                               f"{json.dumps(body)[:300]}")
        time.sleep(backoff * attempt)
    raise RuntimeError("embedding failed after max retries")


class Index:
    def __init__(self, path):
        from rank_bm25 import BM25Okapi
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.chunk_ids = [str(c) for c in payload["chunk_ids"]]
        self.texts = payload["texts"]
        emb = payload["embeddings"]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.embeddings = emb / norms
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.position = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self.text_of = dict(zip(self.chunk_ids, self.texts))

    def dense_ranking(self, query_vec):
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        return [self.chunk_ids[i] for i in np.argsort(-(self.embeddings @ q))]

    def bm25_ranking(self, query):
        return [self.chunk_ids[i] for i in np.argsort(-self.bm25.get_scores(tokenize(query)))]

    def retrieve(self, query, top_k, max_retries=5):
        """(top_ids, top_scores, top_texts, full_ranked_ids) by RRF score."""
        q_vec = embed([query], max_retries=max_retries)[0]
        scores = {}
        for ranking in (self.bm25_ranking(query), self.dense_ranking(q_vec)):
            for rank, cid in enumerate(ranking, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        full = sorted(scores, key=lambda c: -scores[c])
        top = full[:top_k]
        return top, [scores[c] for c in top], [self.text_of[c] for c in top], full

    def in_document_order(self, ids, texts):
        order = sorted(range(len(ids)), key=lambda j: self.position[ids[j]])
        return [ids[j] for j in order], [texts[j] for j in order]


_CACHE = {}


def load_index(path):
    if path not in _CACHE:
        _CACHE[path] = Index(path)
    return _CACHE[path]


# ---------------------------------------------------------------- metrics
def hit_at_k(ids, gold):
    return 1.0 if set(ids) & gold else 0.0


def recall_at_k(ids, gold):
    return len(set(ids) & gold) / len(gold) if gold else 0.0


def mrr(full_ranked, gold):
    for rank, cid in enumerate(full_ranked, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ids, gold, k):
    dcg = sum((1.0 if c in gold else 0.0) / log2(i + 1) for i, c in enumerate(ids[:k], start=1))
    idcg = sum(1.0 / log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0
