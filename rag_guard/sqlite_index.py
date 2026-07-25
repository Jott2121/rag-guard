"""Inverted TF-IDF index on sqlite: same ranking as Retriever, without loading it all.

Why this exists. The JSON index (index.py) stores one forward vector per chunk, so
answering a 5-term query means deserializing every vector in the corpus. Measured on
Jeff's live vault -- 10,926 chunks / 34.2 MB -- that is 2.23s of a 2.56s hook, paid on
every prompt in a cold process.

Storing postings (term -> [(chunk, weight)]) inverts the cost: a query reads only the
rows for its own terms. Measured on the same corpus, cold process, same top-5 in the
same order on 79 real prompts: ~50-200ms.

This is a STORAGE swap, not a retrieval change. The scoring below is a transcription of
Retriever.retrieve() -- raw term counts (not sublinear tf), smoothed idf, cosine over
precomputed norms, doc weight applied after, score rounded to 6dp, ties broken by corpus
order via a stable sort. tests/test_sqlite_index.py asserts equality against Retriever
rather than against hand-written expectations, so the two cannot drift silently.
"""
from __future__ import annotations

import math
import os
import sqlite3

from rag_guard.corpus import build_corpus
from rag_guard.index import fingerprint as corpus_fingerprint
from rag_guard.retriever import _toks

_SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE docs(rowid_ INTEGER PRIMARY KEY, id TEXT, text TEXT, weight REAL, norm REAL);
CREATE TABLE idf(term TEXT PRIMARY KEY, value REAL);
CREATE TABLE postings(term TEXT, doc INTEGER, weight REAL);
"""


class SqliteIndex:
    """Drop-in for Retriever's retrieve(query, k) seam, backed by an inverted index."""

    def __init__(self, path: str):
        self.path = path

    # ---------- build ----------

    def build(self, docs, fingerprint: str) -> None:
        """Rebuild from scratch. Writes to a temp db and renames, so a crash mid-build
        leaves the previous index intact rather than a half-written one."""
        tmp = self.path + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        try:
            con = sqlite3.connect(tmp)
            try:
                con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;" + _SCHEMA)
                self._write(con, docs, fingerprint)
                con.commit()
            finally:
                con.close()
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    @staticmethod
    def _write(con, docs, fingerprint) -> None:
        # Materialize once: build() may be handed a generator, and we need two passes
        # (document frequency, then vectors).
        docs = list(docs)
        token_lists = [_toks(d["text"]) for d in docs]
        n = len(docs) or 1
        df: dict[str, int] = {}
        for toks in token_lists:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}

        doc_rows, posting_rows = [], []
        for i, (doc, toks) in enumerate(zip(docs, token_lists)):
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            vec = {t: c * idf.get(t, 0.0) for t, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            doc_rows.append((i, doc["id"], doc["text"], doc.get("weight", 1.0), norm))
            posting_rows.extend((t, i, w) for t, w in vec.items())

        con.execute("INSERT INTO meta VALUES('fingerprint', ?)", (fingerprint,))
        con.executemany("INSERT INTO docs VALUES(?,?,?,?,?)", doc_rows)
        con.executemany("INSERT INTO idf VALUES(?,?)", idf.items())
        con.executemany("INSERT INTO postings VALUES(?,?,?)", posting_rows)
        con.execute("CREATE INDEX postings_term ON postings(term)")

    # ---------- read ----------

    def fingerprint(self):
        con = self._connect()
        if con is None:
            return None
        try:
            row = con.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
            return row[0] if row else None
        except sqlite3.DatabaseError:
            return None
        finally:
            con.close()

    def _connect(self):
        if not os.path.exists(self.path):
            return None
        try:
            return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        except sqlite3.DatabaseError:
            return None

    def retrieve(self, query, k=3) -> list[dict]:
        qt: dict[str, int] = {}
        for t in _toks(query):
            qt[t] = qt.get(t, 0) + 1
        con = self._connect()
        if con is None:
            return []
        try:
            if not qt:
                # Stopwords-only / empty query carries no signal. Retriever still returns
                # k docs at score 0.0; the callers' MIN_SCORE gate is what discards them.
                return self._zero_padding(con, k, set())
            placeholders = ",".join("?" * len(qt))
            idf = dict(con.execute(
                f"SELECT term, value FROM idf WHERE term IN ({placeholders})", list(qt)))
            qv = {t: c * idf.get(t, 0.0) for t, c in qt.items()}
            qn = math.sqrt(sum(v * v for v in qv.values()))
            if not qn:
                # Every term is out-of-vocabulary: Retriever scores the whole corpus 0.0
                # and still returns k docs in corpus order. Match that.
                return self._zero_padding(con, k, set())

            dots: dict[int, float] = {}
            for term, weight in qv.items():
                if not weight:
                    continue
                for doc, dw in con.execute(
                        "SELECT doc, weight FROM postings WHERE term=?", (term,)):
                    dots[doc] = dots.get(doc, 0.0) + weight * dw

            hits = []
            if dots:
                placeholders = ",".join("?" * len(dots))
                rows = con.execute(
                    f"SELECT rowid_, id, text, weight, norm FROM docs "
                    f"WHERE rowid_ IN ({placeholders}) ORDER BY rowid_", list(dots))
                # Corpus order + a stable sort reproduces Retriever's tie-breaking exactly.
                for doc, did, text, weight, norm in rows:
                    score = (dots[doc] / (qn * norm) * weight) if norm else 0.0
                    hits.append({"id": did, "text": text, "score": round(score, 6)})
            hits.sort(key=lambda h: h["score"], reverse=True)

            scored = [h for h in hits if h["score"] > 0][:k]
            if len(scored) < k:
                scored.extend(self._zero_padding(con, k - len(scored),
                                                 {h["id"] for h in scored}))
        except sqlite3.DatabaseError:
            # Corrupt/truncated cache. Retrieval is advisory, so degrade to silence and
            # let get_sqlite_index() rebuild -- never propagate into a caller's prompt.
            return []
        finally:
            con.close()
        return scored[:k]

    @staticmethod
    def _zero_padding(con, need, exclude):
        """Retriever scores every document and returns k of them, so a query with fewer
        than k real matches comes back padded with score-0.0 docs in corpus order. Callers
        (and the eval harness) compare against that shape, so reproduce it rather than
        quietly returning a shorter list."""
        out = []
        for did, text in con.execute("SELECT id, text FROM docs ORDER BY rowid_"):
            if did in exclude:
                continue
            out.append({"id": did, "text": text, "score": 0.0})
            if len(out) == need:
                break
        return out


def get_sqlite_index(cache_path: str, roots: list[str], *,
                     chunk_chars: int = 800, overlap: int = 100) -> SqliteIndex:
    """Return a current index, rebuilding only when the corpus fingerprint moved."""
    index = SqliteIndex(cache_path)
    current = corpus_fingerprint(roots, chunk_chars=chunk_chars, overlap=overlap)
    # A corrupt or truncated db reports no fingerprint, so it rebuilds rather than
    # serving garbage -- the hook's fail-open promise depends on that.
    if index.fingerprint() != current:
        index.build(build_corpus(roots, chunk_chars=chunk_chars, overlap=overlap), current)
    return index
