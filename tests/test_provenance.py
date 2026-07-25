"""Chunk ids must name the vault a passage came from.

Ids were relative to each corpus root, so a citation read "(CLAUDE.md#2)". Across Jeff's
13 vaults, 232 of 590 files share a basename -- CLAUDE.md appears 14 times, 00-Index.md 7,
Decisions-Log.md 4. The hook tells the agent to prefer those notes while giving it no way
to tell which vault they came from, and measurement put the injected notes' relevance at
38.5%, so this is untraceable *useful* context, not untraceable noise.
"""
import os

from rag_guard.corpus import build_corpus
from rag_guard.index import fingerprint


def _vaults(tmp_path):
    a = tmp_path / "Documents" / "Spring-Bear-2026-Wiki"
    b = tmp_path / "Documents" / "Hunt-621-Wiki"
    mem = tmp_path / "state" / "memory"
    for d in (a, b, mem):
        d.mkdir(parents=True)
    (a / "CLAUDE.md").write_text("spring bear vault guide, black bear over bait")
    (b / "CLAUDE.md").write_text("elk unit 621 vault guide, archery season")
    (mem / "CLAUDE.md").write_text("memory root guide")
    return a, b, mem


def test_ids_are_qualified_by_vault(tmp_path):
    a, b, mem = _vaults(tmp_path)
    ids = {d["id"] for d in build_corpus([str(a), str(b), str(mem)])}
    assert "Spring-Bear-2026-Wiki/CLAUDE.md#0" in ids
    assert "Hunt-621-Wiki/CLAUDE.md#0" in ids
    assert "memory/CLAUDE.md#0" in ids


def test_same_basename_across_vaults_no_longer_collides(tmp_path):
    a, b, mem = _vaults(tmp_path)
    docs = build_corpus([str(a), str(b), str(mem)])
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids)) == 3


def test_a_single_file_root_is_qualified_by_its_directory(tmp_path):
    """~/.claude/CLAUDE.md is a file root, and bare "CLAUDE.md" is the worst offender."""
    d = tmp_path / ".claude"
    d.mkdir()
    f = d / "CLAUDE.md"
    f.write_text("global instructions")
    assert [x["id"] for x in build_corpus([str(f)])] == [".claude/CLAUDE.md#0"]


def test_nested_notes_keep_their_subpath(tmp_path):
    vault = tmp_path / "Documents" / "Agent-Studio-Wiki"
    (vault / "01-Strategy").mkdir(parents=True)
    (vault / "01-Strategy" / "Decisions-Log.md").write_text("decision one")
    ids = [d["id"] for d in build_corpus([str(vault)])]
    assert ids == ["Agent-Studio-Wiki/01-Strategy/Decisions-Log.md#0"]


def test_ids_resolve_back_to_a_real_file(tmp_path):
    """The id has to be reversible or the log and the report cannot join on it."""
    a, b, mem = _vaults(tmp_path)
    roots = [str(a), str(b), str(mem)]
    for doc in build_corpus(roots):
        rel = doc["id"].split("#")[0]
        assert any(os.path.exists(os.path.join(os.path.dirname(
            r if os.path.isdir(r) else os.path.dirname(r)), rel)) for r in roots), doc["id"]


def test_weighting_still_applies_to_the_memory_root(tmp_path):
    """The memory weight keys off the path, not the id -- qualifying ids must not break it."""
    _, _, mem = _vaults(tmp_path)
    assert all(d["weight"] > 1.0 for d in build_corpus([str(mem)]))


def test_fingerprint_changes_with_the_id_schema(tmp_path):
    """An existing cache holds OLD-format ids. Without a schema bump in the fingerprint it
    would be considered current and keep serving unqualified citations forever."""
    from rag_guard import index as index_mod
    a, _, _ = _vaults(tmp_path)
    before = fingerprint([str(a)])
    original = index_mod.CORPUS_SCHEMA
    try:
        index_mod.CORPUS_SCHEMA = original + 1
        assert fingerprint([str(a)]) != before
    finally:
        index_mod.CORPUS_SCHEMA = original
