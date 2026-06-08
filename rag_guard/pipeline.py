"""Guarded RAG pipeline: retrieve -> refuse-if-unsupported -> grounded prompt ->
model -> groundedness check + PII redaction -> a structured, observable result.

Every answer carries a trace (what was retrieved + scores, whether it refused,
whether the answer was grounded) so the system is auditable — the thing that
turns a RAG demo into something you'd run for a client.
"""
from __future__ import annotations

from rag_guard import guard as G
from rag_guard import retriever as R

REFUSAL = "I don't have enough information in the provided sources to answer that."


def build_grounded_prompt(query: str, contexts: list[str]) -> str:
    blocks = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "Answer the question using ONLY the context below. If the context does not "
        "contain the answer, say you don't know — do not use outside knowledge.\n\n"
        f"Context:\n{blocks}\n\nQuestion: {query}\n\nAnswer:"
    )


class RagGuard:
    def __init__(self, retriever: R.Retriever, provider, *, k: int = 3,
                 min_score: float = 0.05, ground_threshold: float = 0.5):
        self.retriever = retriever
        self.provider = provider
        self.k = k
        self.min_score = min_score
        self.ground_threshold = ground_threshold

    def answer(self, query: str) -> dict:
        hits = self.retriever.retrieve(query, self.k)
        trace = {"query": query, "retrieved": [{"id": h["id"], "score": h["score"]} for h in hits]}

        # Guard 1: refuse when retrieval found no real support.
        if G.should_refuse(hits, self.min_score):
            trace["refused"] = True
            return {"answer": REFUSAL, "refused": True, "grounded": None,
                    "support": 0.0, "sources": [], "trace": trace}

        contexts = [h["text"] for h in hits]
        raw = self.provider.complete(build_grounded_prompt(query, contexts))

        # Guard 2: is the answer actually backed by the context?
        g = G.groundedness(raw, contexts, self.ground_threshold)
        # Guard 3: strip PII from whatever the model produced.
        safe = G.redact_pii(raw)

        trace.update({"refused": False, "grounded": g["grounded"], "support": g["support"]})
        return {
            "answer": safe,
            "refused": False,
            "grounded": g["grounded"],
            "support": g["support"],
            "sources": [h["id"] for h in hits],
            "trace": trace,
        }
