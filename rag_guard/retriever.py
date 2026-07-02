"""Pure-stdlib TF-IDF cosine retriever — real RAG retrieval, no heavy deps.

Small and dependency-free so CI is fast and green and there are no API keys.
The `retrieve()` interface is the seam: swap in real embeddings / a vector DB
later without changing the pipeline or the guardrails that depend on the scores.
"""
from __future__ import annotations

import math
import re

_TOKEN = re.compile(r"[a-z0-9]{2,}")

# Function words carry no retrieval signal; counting them creates spurious matches
# (e.g. a query sharing only "is/the/of" with a doc) that would defeat the
# refuse-when-unsupported guard. Drop them at tokenization.
STOPWORDS = {
    "the", "is", "are", "was", "were", "be", "been", "being", "and", "or", "but",
    "of", "to", "in", "on", "at", "by", "for", "with", "as", "from", "into", "about",
    "a", "an", "this", "that", "these", "those", "it", "its", "you", "your", "we",
    "our", "they", "them", "i", "me", "my", "he", "she", "his", "her",
    "what", "which", "who", "whom", "how", "when", "where", "why", "do", "does",
    "did", "can", "could", "will", "would", "should", "may", "might", "have", "has",
    "had", "not", "no", "if", "then", "so", "than", "too", "very", "just", "any",
}


def _toks(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOPWORDS]


class Retriever:
    def __init__(self, docs: list[dict]):
        self.docs = list(docs)
        token_lists = [_toks(d["text"]) for d in self.docs]
        n = len(self.docs) or 1
        df: dict[str, int] = {}
        for toks in token_lists:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        # smoothed idf, always positive
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self._vecs = [self._vectorize(toks) for toks in token_lists]
        self._norms = [self._norm(v) for v in self._vecs]

    def _vectorize(self, toks: list[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        return {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}

    @staticmethod
    def _norm(vec):
        import math
        return math.sqrt(sum(v * v for v in vec.values()))

    @classmethod
    def from_index(cls, docs, idf, vecs, norms):
        self = cls.__new__(cls)
        self.docs, self.idf, self._vecs, self._norms = docs, idf, vecs, norms
        return self

    def index_state(self):
        return {"docs": self.docs, "idf": self.idf, "vecs": self._vecs, "norms": self._norms}

    def retrieve(self, query, k=3):
        import math
        qv = self._vectorize(_toks(query))
        qn = math.sqrt(sum(v * v for v in qv.values()))
        scored = []
        for i, d in enumerate(self.docs):
            dn = self._norms[i]
            if qn and dn:
                common = set(qv) & set(self._vecs[i])
                score = sum(qv[t] * self._vecs[i][t] for t in common) / (qn * dn)
            else:
                score = 0.0
            score *= d.get("weight", 1.0)
            scored.append({"id": d["id"], "text": d["text"], "score": round(score, 6)})
        scored.sort(key=lambda h: h["score"], reverse=True)
        return scored[:k]
