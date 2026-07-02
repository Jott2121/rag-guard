from rag_guard import reindex

def test_reindex_builds_cache(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("hello world"); (tmp_path / "b.md").write_text("more text")
    cache = tmp_path / "i.json"
    monkeypatch.setattr("rag_guard.config.default_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr("rag_guard.config.cache_path", lambda: str(cache))
    assert reindex.reindex() >= 1 and cache.exists()
