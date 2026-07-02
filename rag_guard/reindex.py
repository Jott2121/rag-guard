"""Force a full index rebuild — used by the nightly launchd backstop."""
from __future__ import annotations

import os

from rag_guard import config
from rag_guard.index import get_index


def reindex() -> int:
    cache = config.cache_path()
    try:
        os.remove(cache)
    except OSError:
        pass
    return len(get_index(cache, config.default_roots()).docs)


if __name__ == "__main__":
    print(reindex())
