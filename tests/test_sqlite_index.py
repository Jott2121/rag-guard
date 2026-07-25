"""The sqlite index must be a pure storage swap: identical ranking, far less load cost.

The JSON index deserializes every chunk vector to answer a 5-term query -- 2.23s of a
2.56s hook on Jeff's 10.9k-chunk corpus. This stores an inverted index so a query touches
only the postings for its own terms. Correctness bar: byte-identical results to Retriever.
"""
import json
import os

import pytest

from rag_guard.corpus import build_corpus
from rag_guard.retriever import Retriever
from rag_guard.sqlite_index import SqliteIndex, get_sqlite_index

DOCS = [
    {"id": "a#0", "text": "elk hunting in the pintlers during september archery season", "weight": 1.0},
    {"id": "b#0", "text": "jeff prefers plain text deliverables with zero markdown", "weight": 1.15},
    {"id": "c#0", "text": "the trading bot backtest showed no durable edge", "weight": 1.0},
    {"id": "d#0", "text": "plain text deliverables land on the desktop", "weight": 1.0},
    {"id": "e#0", "text": "septembers archery elk tactics for public land", "weight": 1.0},
]
QUERIES = ["plain text deliverables", "elk archery september", "backtest edge",
           "markdown desktop", "nothing matches this query at all", "jeff"]


@pytest.fixture
def built(tmp_path):
    idx = SqliteIndex(str(tmp_path / "i.sqlite"))
    idx.build(DOCS, fingerprint="fp1")
    return idx


def test_matches_retriever_exactly(built):
    """The whole justification for the swap: same math, different storage."""
    ref = Retriever(DOCS)
    for q in QUERIES:
        for k in (1, 3, 5):
            assert built.retrieve(q, k) == ref.retrieve(q, k), (q, k)


def test_document_weight_is_applied(tmp_path):
    """memory/ notes carry weight 1.15; dropping it would silently change ranking."""
    weighted = SqliteIndex(str(tmp_path / "w.sqlite"))
    weighted.build(DOCS, fingerprint="fp")
    plain = SqliteIndex(str(tmp_path / "p.sqlite"))
    plain.build([{**d, "weight": 1.0} for d in DOCS], fingerprint="fp")

    def score(idx):
        return {h["id"]: h["score"] for h in idx.retrieve("markdown", 5)}["b#0"]

    assert score(weighted) == pytest.approx(score(plain) * 1.15)


def test_signal_free_queries_match_retriever_shape(built):
    """Empty / stopword-only / all-OOV queries carry no signal. Retriever still returns k
    docs at score 0.0 (the caller\'s MIN_SCORE gate is what discards them), so the sqlite
    index must return the same shape rather than a silently shorter list."""
    ref = Retriever(DOCS)
    for q in ("", "what is the of to", "zzz qqq wwww"):
        assert built.retrieve(q, 5) == ref.retrieve(q, 5), q
        assert all(h["score"] == 0.0 for h in built.retrieve(q, 5)), q


def test_scores_are_rounded_like_the_retriever(built):
    for h in built.retrieve("plain text deliverables", 5):
        assert h["score"] == round(h["score"], 6)


def test_fingerprint_roundtrips(built):
    assert built.fingerprint() == "fp1"


def test_missing_database_reports_no_fingerprint(tmp_path):
    assert SqliteIndex(str(tmp_path / "absent.sqlite")).fingerprint() is None


def test_rebuild_replaces_old_content(built):
    built.build([{"id": "z#0", "text": "entirely different content about boats", "weight": 1.0}],
                fingerprint="fp2")
    assert built.fingerprint() == "fp2"
    assert [h["id"] for h in built.retrieve("boats", 5)] == ["z#0"]
    assert [h for h in built.retrieve("pintlers", 5) if h["score"] > 0] == []


def test_build_is_atomic_on_failure(tmp_path):
    """A crash mid-build must not leave a half-written index that answers wrongly."""
    path = str(tmp_path / "i.sqlite")
    idx = SqliteIndex(path)
    idx.build(DOCS, fingerprint="good")

    class Boom(list):
        def __iter__(self):
            yield {"id": "x#0", "text": "partial", "weight": 1.0}
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        idx.build(Boom(), fingerprint="bad")
    fresh = SqliteIndex(path)
    assert fresh.fingerprint() == "good"
    assert fresh.retrieve("pintlers", 5)


def test_get_sqlite_index_builds_then_reuses(tmp_path):
    root = tmp_path / "memory"
    root.mkdir()
    (root / "n.md").write_text("elk hunting in the pintlers during september")
    cache = str(tmp_path / "i.sqlite")

    first = get_sqlite_index(cache, [str(root)])
    assert first.retrieve("pintlers elk", 3)
    stamp = os.path.getmtime(cache)

    second = get_sqlite_index(cache, [str(root)])
    assert second.retrieve("pintlers elk", 3) == first.retrieve("pintlers elk", 3)
    assert os.path.getmtime(cache) == stamp, "unchanged corpus must not trigger a rebuild"


def test_get_sqlite_index_rebuilds_when_corpus_changes(tmp_path):
    root = tmp_path / "memory"
    root.mkdir()
    note = root / "n.md"
    note.write_text("elk hunting in the pintlers")
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(root)])

    note.write_text("boat maintenance and outboard motors")
    os.utime(note, (note.stat().st_atime, note.stat().st_mtime + 10))
    rebuilt = get_sqlite_index(cache, [str(root)])
    assert [h["id"] for h in rebuilt.retrieve("outboard motors", 3) if h["score"] > 0]
    assert [h for h in rebuilt.retrieve("pintlers", 3) if h["score"] > 0] == []


def test_matches_retriever_on_real_chunked_markdown(tmp_path):
    """End-to-end over build_corpus output, including multi-chunk notes and weights."""
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "long.md").write_text("elk season planning. " * 200)
    (mem / "short.md").write_text("jeff prefers plain text deliverables")
    docs = build_corpus([str(mem)])
    assert len(docs) > 2
    idx = SqliteIndex(str(tmp_path / "i.sqlite"))
    idx.build(docs, fingerprint="fp")
    ref = Retriever(docs)
    for q in ("elk season planning", "plain text deliverables", "jeff"):
        assert idx.retrieve(q, 5) == ref.retrieve(q, 5), q


def test_index_file_is_smaller_than_the_json_equivalent(tmp_path):
    """Sanity: we are not trading 2.2s of parse for a pathological on-disk blowup."""
    mem = tmp_path / "memory"
    mem.mkdir()
    for i in range(40):
        (mem / f"n{i}.md").write_text(f"note {i} about elk hunting and trading bots " * 40)
    docs = build_corpus([str(mem)])
    idx = SqliteIndex(str(tmp_path / "i.sqlite"))
    idx.build(docs, fingerprint="fp")
    ref = Retriever(docs)
    json_size = len(json.dumps({"fingerprint": "fp", **ref.index_state()}))
    assert os.path.getsize(str(tmp_path / "i.sqlite")) < json_size * 3


def test_corrupt_index_reports_no_fingerprint_and_retrieves_nothing(tmp_path):
    """A truncated/garbage cache must degrade to silence, never to wrong passages."""
    path = tmp_path / "i.sqlite"
    path.write_bytes(b"this is not a sqlite database at all, not even close")
    idx = SqliteIndex(str(path))
    assert idx.fingerprint() is None
    assert idx.retrieve("elk hunting", 5) == []


def test_get_sqlite_index_heals_a_corrupt_cache(tmp_path):
    """Because a corrupt cache reports no fingerprint, the next call rebuilds it."""
    root = tmp_path / "memory"
    root.mkdir()
    (root / "n.md").write_text("elk hunting in the pintlers during september")
    cache = tmp_path / "i.sqlite"
    cache.write_bytes(b"garbage")
    idx = get_sqlite_index(str(cache), [str(root)])
    assert [h for h in idx.retrieve("pintlers elk", 3) if h["score"] > 0]


def test_retrieve_on_a_missing_index_is_silent(tmp_path):
    assert SqliteIndex(str(tmp_path / "absent.sqlite")).retrieve("anything", 5) == []
