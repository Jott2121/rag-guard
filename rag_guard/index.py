"""Persist the TF-IDF index; rebuild only when the corpus fingerprint changes."""
from __future__ import annotations

import hashlib
import json
import os

from rag_guard.corpus import build_corpus
from rag_guard.retriever import Retriever

_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}


def _walk_md(root):
    if os.path.isfile(root):
        if root.endswith(".md"):
            yield root
        return
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in _SKIP_DIRS]
        for name in sorted(fn):
            if name.endswith(".md"):
                yield os.path.join(dp, name)


# Bump when the shape of a chunk id or its text changes. A cached index whose ids are in
# an old format is not "current" no matter what the corpus mtimes say, and without this in
# the fingerprint it would keep being served.
CORPUS_SCHEMA = 2


def fingerprint(roots, *, chunk_chars=800, overlap=100) -> str:
    h = hashlib.sha256()
    h.update(f"v{CORPUS_SCHEMA}:{chunk_chars}:{overlap}:".encode())
    for root in sorted(roots):
        for path in _walk_md(root):
            try:
                st = os.stat(path)
            except OSError:
                continue
            h.update(f"{path}:{int(st.st_mtime)}:{st.st_size};".encode())
    return h.hexdigest()


def save_index(retriever, fp, path):
    payload = {"fingerprint": fp, **retriever.index_state()}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def load_index(path):
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    fp = payload.pop("fingerprint", None)
    try:
        return Retriever.from_index(payload["docs"], payload["idf"],
                                    payload["vecs"], payload["norms"]), fp
    except KeyError:
        return None


def get_index(cache_path, roots, *, chunk_chars=800, overlap=100) -> Retriever:
    current = fingerprint(roots, chunk_chars=chunk_chars, overlap=overlap)
    cached = load_index(cache_path)
    if cached is not None and cached[1] == current:
        return cached[0]
    retriever = Retriever(build_corpus(roots, chunk_chars=chunk_chars, overlap=overlap))
    save_index(retriever, current, cache_path)
    return retriever
