from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_model = None


def _model_instance():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def bm25_scores(query: str, docs: List[str]) -> List[float]:
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25.get_scores(query.lower().split()).tolist()


def embed(texts: List[str]) -> np.ndarray:
    m = _model_instance()
    return np.asarray(m.encode(texts, normalize_embeddings=True))


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def hybrid_rank(query: str, items: List[Dict]) -> List[Tuple[int, float]]:
    texts = [f"{i.get('title','')} {i.get('abstract') or ''}" for i in items]
    bm = np.array(bm25_scores(query, texts))
    embs = embed([query] + texts)
    qv, dv = embs[0:1], embs[1:]
    cos = cosine(qv, dv).flatten()
    bm_r = np.argsort(np.argsort(-bm)) + 1
    cs_r = np.argsort(np.argsort(-cos)) + 1
    score = 1 / bm_r + 1 / cs_r
    ranked = sorted(list(enumerate(score)), key=lambda x: -x[1])
    return ranked


def deduplicate(items: List[Dict]) -> List[Dict]:
    out = []
    seen = []
    for it in items:
        title = (it.get("title") or "").lower()
        doi = (it.get("doi") or "").lower()
        dup = False
        for _, t, d in seen:
            if doi and d and doi == d: dup = True
            if fuzz.token_set_ratio(title, t) > 92: dup = True
        if not dup:
            seen.append((len(out), title, doi))
            out.append(it)
    return out
