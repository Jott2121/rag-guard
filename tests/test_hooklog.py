"""The hook's observability log: append-only, opt-in, and incapable of breaking the hook.

This exists to answer one question with data instead of intuition -- are injected passages
ever actually used? -- before anyone touches MIN_SCORE / HOOK_MIN_OVERLAP. It records
SILENT decisions too, because underfiring is the failure mode you cannot see.
"""
import json
import os

from rag_guard import hooklog


def test_disabled_by_default(tmp_path):
    """It records prompts, so a public install must not start logging silently."""
    path = tmp_path / "fires.jsonl"
    assert hooklog.enabled(env={}) is False
    assert hooklog.log_event({"a": 1}, path=str(path), env={}) is False
    assert not path.exists()


def test_enabled_by_env(tmp_path):
    path = tmp_path / "fires.jsonl"
    assert hooklog.enabled(env={"RAG_GUARD_HOOK_LOG": "1"}) is True
    assert hooklog.log_event({"a": 1}, path=str(path), env={"RAG_GUARD_HOOK_LOG": "1"}) is True
    assert json.loads(path.read_text())["a"] == 1


def _write(path, event, **kw):
    return hooklog.log_event(event, path=str(path), env={"RAG_GUARD_HOOK_LOG": "1"}, **kw)


def test_appends_one_json_object_per_line(tmp_path):
    path = tmp_path / "fires.jsonl"
    for i in range(3):
        _write(path, {"i": i})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["i"] for r in rows] == [0, 1, 2]


def test_stamps_a_timestamp_when_absent(tmp_path):
    path = tmp_path / "fires.jsonl"
    _write(path, {"i": 0})
    assert "ts" in json.loads(path.read_text())


def test_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "deeper" / "fires.jsonl"
    assert _write(path, {"i": 0}) is True
    assert path.exists()


def test_log_file_is_owner_only(tmp_path):
    """It contains verbatim prompts. Other local users have no business reading it."""
    path = tmp_path / "fires.jsonl"
    _write(path, {"i": 0})
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"


def test_rotates_instead_of_growing_without_bound(tmp_path):
    path = tmp_path / "fires.jsonl"
    for i in range(50):
        _write(path, {"i": i, "pad": "x" * 200}, max_bytes=2000)
    assert os.path.getsize(path) <= 2000 + 400
    assert (tmp_path / "fires.jsonl.1").exists(), "previous window must be kept, not dropped"


def test_rotation_keeps_only_one_previous_window(tmp_path):
    path = tmp_path / "fires.jsonl"
    for i in range(200):
        _write(path, {"i": i, "pad": "x" * 200}, max_bytes=1000)
    assert not (tmp_path / "fires.jsonl.2").exists()


def test_never_raises_on_an_unwritable_path(tmp_path):
    """Fail-silent is the whole contract: instrumentation must not break grounding."""
    blocked = tmp_path / "ro"
    blocked.mkdir()
    os.chmod(blocked, 0o500)
    try:
        assert _write(blocked / "fires.jsonl", {"i": 0}) is False
    finally:
        os.chmod(blocked, 0o700)


def test_never_raises_on_unserializable_content(tmp_path):
    path = tmp_path / "fires.jsonl"
    assert _write(path, {"bad": {1, 2, 3}}) is False
    assert not path.exists() or path.read_text() == ""


def test_reader_skips_corrupt_lines(tmp_path):
    path = tmp_path / "fires.jsonl"
    _write(path, {"i": 0})
    with open(path, "a") as f:
        f.write("{ this is not json\n")
    _write(path, {"i": 1})
    assert [r["i"] for r in hooklog.read_events(str(path))] == [0, 1]


def test_reader_on_missing_file_is_empty(tmp_path):
    assert hooklog.read_events(str(tmp_path / "absent.jsonl")) == []


def test_reader_includes_the_rotated_window(tmp_path):
    path = tmp_path / "fires.jsonl"
    for i in range(50):
        _write(path, {"i": i, "pad": "x" * 200}, max_bytes=2000)
    seen = [r["i"] for r in hooklog.read_events(str(path))]
    assert seen == sorted(seen) and len(seen) > 10, "rotated events must not vanish"
