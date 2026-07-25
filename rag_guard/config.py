"""Corpus roots, cache path, and thresholds."""
from __future__ import annotations

import glob
import os

MIN_SCORE = 0.05
MAX_WEB_SOURCES = 6
# The hook stays silent unless the top note shares at least this many distinct
# content-words with the prompt. Raw TF-IDF score alone doesn't separate relevant
# from tangential (short queries spike on a single shared rare word); requiring
# >=2 shared content-words is what actually gates noise. See probe in git history.
HOOK_MIN_OVERLAP = 2


def _cache_dir() -> str:
    base = os.path.expanduser("~/.cache/rag-guard")
    os.makedirs(base, exist_ok=True)
    return base


def cache_path() -> str:
    return os.path.join(_cache_dir(), "index.json")


def sqlite_cache_path() -> str:
    """Inverted-index cache. Separate file from index.json so the two backends can
    coexist and RAG_GUARD_BACKEND=json stays an instant rollback."""
    return os.path.join(_cache_dir(), "index.sqlite")


def default_roots() -> list[str]:
    override = os.environ.get("RAG_GUARD_ROOTS")
    if override:
        candidates = override.split(":")
    else:
        home = os.path.expanduser("~")
        candidates = [
            os.path.join(home, ".claude/projects/-Users-jeffreyotterson/memory"),
            *glob.glob(os.path.join(home, "Documents/*-Wiki")),
            os.path.join(home, ".claude/CLAUDE.md"),
        ]
    return [c for c in candidates if os.path.exists(c)]
