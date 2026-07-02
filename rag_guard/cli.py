"""stdin/argv -> cached retrieval -> JSON stdout. Retrieval-only; no model call."""
from __future__ import annotations

import argparse
import json
import sys

from rag_guard import config, service


def run(argv, stdin_text) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("query", nargs="*")
    args = ap.parse_args(argv)
    q = " ".join(args.query) or (stdin_text or "").strip()
    roots = args.roots.split(":") if args.roots else None
    hits = service.query(q, args.k, roots=roots, cache=args.cache)
    support = max((h["score"] for h in hits), default=0.0)
    return {"query": q, "hits": hits, "support": round(support, 6),
            "grounded": support >= config.MIN_SCORE}


def main():
    stdin_text = "" if sys.stdin.isatty() else sys.stdin.read()
    print(json.dumps(run(sys.argv[1:], stdin_text)))


if __name__ == "__main__":
    main()
