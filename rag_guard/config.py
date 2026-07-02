"""Corpus roots, cache path, and thresholds."""
from __future__ import annotations

import glob
import os

MIN_SCORE = 0.05
MAX_WEB_SOURCES = 6


def cache_path() -> str:
    base = os.path.expanduser("~/.cache/rag-guard")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "index.json")


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
