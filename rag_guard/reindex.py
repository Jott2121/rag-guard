"""Force a full index rebuild — used by the nightly launchd backstop.

Rebuilding is the one expensive operation left: the fingerprint covers the whole corpus
and idf is global, so a single edited note invalidates every vector. Whoever triggers a
rebuild pays it (~13-15s), and if that is the hook, it is paid inline on someone's prompt.
Running this on a schedule is what keeps that cost off the interactive path.

Rebuilds BOTH backends by default so `RAG_GUARD_BACKEND=json` stays a working rollback.
"""
from __future__ import annotations

import os

from rag_guard import config
from rag_guard.index import get_index
from rag_guard.sqlite_index import get_sqlite_index


def _drop(path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def reindex(*, backends=("sqlite", "json")) -> int:
    """Rebuild the requested caches from scratch. Returns the chunk count."""
    roots = config.default_roots()
    count = 0
    if "json" in backends:
        cache = config.cache_path()
        _drop(cache)
        count = len(get_index(cache, roots).docs)
    if "sqlite" in backends:
        cache = config.sqlite_cache_path()
        _drop(cache)
        con = get_sqlite_index(cache, roots)._connect()
        if con is not None:
            try:
                count = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0] or count
            finally:
                con.close()
    return count


if __name__ == "__main__":
    print(reindex())
