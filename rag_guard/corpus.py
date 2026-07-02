"""Build a retrievable corpus of {id,text,source,weight} chunks from markdown files."""
from __future__ import annotations

import os

DEFAULT_EXCLUDE = (".aiff", ".m4a", ".mp3", ".wav", ".pdf", ".png", ".jpg",
                   ".jpeg", ".gif", ".gpx", ".canvas", ".zip")
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
MEMORY_WEIGHT = 1.15


def chunk_text(text: str, *, chunk_chars: int = 800, overlap: int = 100) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if text else []
    step = max(1, chunk_chars - overlap)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_chars]
        if chunk:
            chunks.append(chunk)
        if start + chunk_chars >= len(text):
            break
    return chunks


def _weight_for(path: str) -> float:
    return MEMORY_WEIGHT if (os.sep + "memory" + os.sep) in (path + os.sep) or \
        os.path.basename(os.path.dirname(path)) == "memory" else 1.0


def build_corpus(roots: list[str], *, chunk_chars: int = 800, overlap: int = 100,
                 exclude_suffixes: tuple[str, ...] = DEFAULT_EXCLUDE) -> list[dict]:
    docs: list[dict] = []
    for root in roots:
        if os.path.isfile(root):
            _add_file(root, root, docs, chunk_chars, overlap, exclude_suffixes)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in sorted(filenames):
                if name.endswith(".md"):
                    _add_file(os.path.join(dirpath, name), root, docs,
                              chunk_chars, overlap, exclude_suffixes)
    return docs


def _add_file(path, source, docs, chunk_chars, overlap, exclude_suffixes):
    if path.endswith(exclude_suffixes):
        return
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return
    base = source if os.path.isdir(source) else os.path.dirname(source)
    rel = os.path.relpath(path, base)
    weight = _weight_for(path)
    for i, chunk in enumerate(chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)):
        docs.append({"id": f"{rel}#{i}", "text": chunk, "source": source, "weight": weight})
