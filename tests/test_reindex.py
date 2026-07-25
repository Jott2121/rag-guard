from rag_guard import reindex

def test_reindex_builds_cache(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("hello world"); (tmp_path / "b.md").write_text("more text")
    cache = tmp_path / "i.json"
    monkeypatch.setattr("rag_guard.config.default_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr("rag_guard.config.cache_path", lambda: str(cache))
    assert reindex.reindex() >= 1 and cache.exists()


def test_reindex_builds_both_backends(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("elk hunting in the pintlers")
    jsn, sql = tmp_path / "i.json", tmp_path / "i.sqlite"
    monkeypatch.setattr("rag_guard.config.default_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr("rag_guard.config.cache_path", lambda: str(jsn))
    monkeypatch.setattr("rag_guard.config.sqlite_cache_path", lambda: str(sql))
    assert reindex.reindex() >= 1
    assert jsn.exists() and sql.exists(), "rollback backend must stay warm too"


def test_reindex_can_target_one_backend(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("elk hunting in the pintlers")
    jsn, sql = tmp_path / "i.json", tmp_path / "i.sqlite"
    monkeypatch.setattr("rag_guard.config.default_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr("rag_guard.config.cache_path", lambda: str(jsn))
    monkeypatch.setattr("rag_guard.config.sqlite_cache_path", lambda: str(sql))
    assert reindex.reindex(backends=("sqlite",)) >= 1
    assert sql.exists() and not jsn.exists()
