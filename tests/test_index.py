import time
from rag_guard.index import fingerprint, get_index

def _seed(tmp_path):
    (tmp_path / "a.md").write_text("shipping takes three days")
    (tmp_path / "b.md").write_text("returns within thirty days")

def test_get_index_builds_then_caches(tmp_path):
    _seed(tmp_path); cache = tmp_path / "idx.json"
    r1 = get_index(str(cache), [str(tmp_path)])
    assert cache.exists() and r1.retrieve("shipping")[0]["score"] > 0
    assert get_index(str(cache), [str(tmp_path)]).retrieve("shipping") == r1.retrieve("shipping")

def test_fingerprint_changes_on_file_and_chunkparams(tmp_path):
    _seed(tmp_path)
    fp1 = fingerprint([str(tmp_path)])
    time.sleep(0.01); (tmp_path / "a.md").write_text("longer different content here")
    assert fingerprint([str(tmp_path)]) != fp1
    assert fingerprint([str(tmp_path)], chunk_chars=400) != fingerprint([str(tmp_path)])

def test_get_index_rebuilds_on_change(tmp_path):
    _seed(tmp_path); cache = tmp_path / "idx.json"
    get_index(str(cache), [str(tmp_path)])
    (tmp_path / "a.md").write_text("bravo replacement content indeed")
    assert get_index(str(cache), [str(tmp_path)]).retrieve("bravo")[0]["score"] > 0

def test_load_index_returns_none_on_corrupt(tmp_path):
    from rag_guard.index import load_index
    p = tmp_path / "bad.json"; p.write_text("{not json")
    assert load_index(str(p)) is None
