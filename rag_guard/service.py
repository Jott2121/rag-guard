"""Warm, importable retrieval singleton for in-process callers (Bow)."""
from __future__ import annotations

from rag_guard import config
from rag_guard.index import fingerprint, get_index

_SINGLETON = None
_FP = None


def reset() -> None:
    global _SINGLETON, _FP
    _SINGLETON, _FP = None, None


def query(text, k=5, *, roots=None, cache=None) -> list[dict]:
    global _SINGLETON, _FP
    roots = roots or config.default_roots()
    cache = cache or config.cache_path()
    fp = fingerprint(roots)
    if _SINGLETON is None or _FP != fp:
        _SINGLETON = get_index(cache, roots)
        _FP = fp
    return _SINGLETON.retrieve(text, k)
