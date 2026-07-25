"""Warm, importable retrieval singleton for in-process callers (Bow) and the hook.

The hook is always a COLD process, so "warm singleton" does nothing for it -- what
dominates there is how much of the index has to be read to answer one query. The sqlite
backend reads only the postings for the query's own terms; the JSON backend deserializes
every chunk vector (measured: 2.23s of a 2.56s hook on the live vault). Both produce
identical rankings, so the backend choice is a pure cost decision.

Set RAG_GUARD_BACKEND=json to fall back."""
from __future__ import annotations

import os

from rag_guard import config
from rag_guard.index import fingerprint, get_index
from rag_guard.sqlite_index import get_sqlite_index

_SINGLETON = None
_FP = None


def reset() -> None:
    global _SINGLETON, _FP
    _SINGLETON, _FP = None, None


def _use_sqlite() -> bool:
    return os.environ.get("RAG_GUARD_BACKEND", "sqlite").lower() != "json"


def query(text, k=5, *, roots=None, cache=None) -> list[dict]:
    global _SINGLETON, _FP
    roots = roots or config.default_roots()
    fp = fingerprint(roots)
    if _SINGLETON is None or _FP != fp:
        if _use_sqlite():
            _SINGLETON = get_sqlite_index(cache or config.sqlite_cache_path(), roots)
        else:
            _SINGLETON = get_index(cache or config.cache_path(), roots)
        _FP = fp
    return _SINGLETON.retrieve(text, k)
