from rag_guard import config

def test_cache_path_and_constants():
    assert config.cache_path().endswith("index.json")
    assert config.MIN_SCORE > 0 and config.MAX_WEB_SOURCES >= 2

def test_default_roots_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_GUARD_ROOTS", str(tmp_path))
    (tmp_path / "x.md").write_text("hi")
    assert str(tmp_path) in config.default_roots()
