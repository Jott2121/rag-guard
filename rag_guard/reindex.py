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
from rag_guard.corpus import build_corpus
from rag_guard.index import fingerprint as corpus_fingerprint, save_index
from rag_guard.retriever import Retriever
from rag_guard import rebuild_health
from rag_guard.sqlite_index import SqliteIndex


def reindex(*, backends=("sqlite", "json"), sqlite_cache=None, json_cache=None) -> int:
    """Rebuild the requested caches from scratch. Returns the chunk count.

    Cache paths are explicit parameters, not just config defaults: the background rebuild
    must target the SAME cache its caller is serving from, or it refreshes a different
    file and the caller stays stale forever.

    NEVER deletes the existing cache first. An earlier version did, and it opened a
    ~20-second window where the file was absent: a hook arriving mid-rebuild found no
    index to serve stale, so it fell back to a blocking rebuild of its own -- turning the
    stall this was built to remove into a worse one. Both backends already write to a temp
    file and rename, so a forced rebuild is atomic without any unlink.
    """
    roots = config.default_roots()
    current = corpus_fingerprint(roots)
    count = 0
    if "json" in backends:
        cache = json_cache or config.cache_path()
        retriever = Retriever(build_corpus(roots))
        save_index(retriever, current, cache)
        count = len(retriever.docs)
    if "sqlite" in backends:
        cache = sqlite_cache or config.sqlite_cache_path()
        docs = build_corpus(roots)
        SqliteIndex(cache).build(docs, current)
        count = len(docs) or count
    return count


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild the rag-guard index caches.")
    ap.add_argument("--backends", default="sqlite,json",
                    help="comma-separated: sqlite, json (default both)")
    ap.add_argument("--sqlite-cache", default=None)
    ap.add_argument("--json-cache", default=None)
    args = ap.parse_args(argv)
    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    cache = args.sqlite_cache or config.sqlite_cache_path()
    try:
        count = reindex(backends=backends, sqlite_cache=args.sqlite_cache,
                        json_cache=args.json_cache)
        # This process is detached with stderr to DEVNULL, so a traceback goes nowhere.
        # The marker is the only place a caller can learn the rebuild is broken.
        rebuild_health.record_success(cache, chunks=count)
        return count
    except Exception as exc:
        rebuild_health.record_failure(cache, f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        # The background path claims a lock before spawning us; release it whether the
        # rebuild succeeded or died, or staleness wedges until the lock ages out.
        held = os.environ.get("RAG_GUARD_LOCK_HELD")
        if held:
            from rag_guard.sqlite_index import release_rebuild_lock
            release_rebuild_lock(held)


if __name__ == "__main__":
    print(main())
