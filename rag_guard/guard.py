"""Guardrails — refuse-when-unsupported, groundedness, and PII redaction.

These are what separate a RAG demo from a RAG system you'd put in front of a
client: the model is not allowed to answer from nothing, its answer is checked
against the retrieved context, and PII is stripped from the output. Deterministic
and stdlib so each guard is unit-testable; the groundedness proxy is swappable
for an NLI/LLM judge behind the same interface.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]{2,}")
_STOP = {
    "the", "and", "for", "are", "was", "you", "your", "our", "with", "that", "this",
    "from", "have", "has", "can", "will", "any", "all", "out", "use", "via", "per",
    "is", "to", "of", "in", "on", "at", "it", "or", "be", "we", "a", "an", "as", "by",
    "takes", "take", "made", "business", "days",
}

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP}


def should_refuse(hits: list[dict], min_score: float = 0.05) -> bool:
    """Refuse to answer when retrieval found no real support (top score too low)."""
    if not hits:
        return True
    return max(h.get("score", 0.0) for h in hits) < min_score


def groundedness(answer: str, contexts: list[str], threshold: float = 0.5) -> dict:
    """What fraction of the answer's content words are supported by the context.
    `grounded` is support >= threshold. (Lexical proxy for an NLI/LLM judge.)"""
    ans = _content_tokens(answer)
    if not ans:
        return {"grounded": False, "support": 0.0}
    ctx = set()
    for c in contexts:
        ctx |= _content_tokens(c)
    support = len(ans & ctx) / len(ans)
    return {"grounded": support >= threshold, "support": round(support, 4)}


def redact_pii(text: str) -> str:
    """Mask emails, phone numbers, SSNs, and card-like numbers in an output."""
    text = _EMAIL.sub("[redacted-email]", text)
    text = _SSN.sub("[redacted-ssn]", text)
    text = _PHONE.sub("[redacted-phone]", text)
    text = _CARD.sub("[redacted-card]", text)
    return text
