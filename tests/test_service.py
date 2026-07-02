from rag_guard import service

def test_query_returns_hits(tmp_path):
    (tmp_path / "a.md").write_text("shipping takes three days")
    (tmp_path / "b.md").write_text("returns within thirty days")
    service.reset()
    hits = service.query("shipping", roots=[str(tmp_path)], cache=str(tmp_path / "i.json"))
    assert hits and hits[0]["score"] > 0

def test_reset_clears_singleton(tmp_path):
    (tmp_path / "a.md").write_text("alpha"); (tmp_path / "b.md").write_text("beta")
    service.reset()
    service.query("alpha", roots=[str(tmp_path)], cache=str(tmp_path / "i.json"))
    assert service._SINGLETON is not None
    service.reset()
    assert service._SINGLETON is None
