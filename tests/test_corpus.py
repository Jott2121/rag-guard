from rag_guard.corpus import chunk_text, build_corpus

def test_chunk_respects_size_and_overlap():
    text = "abcdefghij" * 20
    chunks = chunk_text(text, chunk_chars=80, overlap=20)
    assert all(len(c) <= 80 for c in chunks)
    assert len(chunks) >= 3
    assert chunks[0][-20:] == chunks[1][:20]

def test_chunk_short_text_single_chunk():
    assert chunk_text("hello world", chunk_chars=800, overlap=100) == ["hello world"]

def test_build_corpus_reads_markdown_and_skips_excluded(tmp_path):
    (tmp_path / "a.md").write_text("shipping takes 3 days")
    (tmp_path / "b.aiff").write_bytes(b"\x00\x01")
    (tmp_path / "notes.txt").write_text("ignore me")
    docs = build_corpus([str(tmp_path)])
    ids = [d["id"] for d in docs]
    assert any("a.md#0" in i for i in ids)
    assert all(".aiff" not in i and "notes.txt" not in i for i in ids)
    assert docs[0]["text"] == "shipping takes 3 days"
    assert docs[0]["weight"] == 1.0

def test_memory_source_gets_higher_weight(tmp_path):
    mem = tmp_path / "memory"; mem.mkdir()
    (mem / "fact.md").write_text("Jeff prefers plain text")
    docs = build_corpus([str(mem)])
    assert docs[0]["weight"] == 1.15
