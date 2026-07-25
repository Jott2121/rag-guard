"""Serving a stale index beats stalling a human for 15 seconds.

Rebuilding is global -- idf is corpus-wide, so one edited note invalidates every vector,
and the rebuild costs 12-16s. Measured on the live vault, that cost lands INLINE on the
next prompt after any memory or wiki write, which is most sessions. Serving the slightly
stale index and rebuilding behind it removes the stall unconditionally.

The dangerous part is not the staleness, it is the subprocess: this runs inside a hook
that prints JSON to stdout, on a machine running several autonomous agents.
"""
import os
import subprocess
import sys
import time

import pytest

from rag_guard import config
from rag_guard.sqlite_index import SqliteIndex, get_sqlite_index


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "memory"
    root.mkdir()
    (root / "a.md").write_text("elk hunting in the pintlers during september archery")
    return root


def _touch(path, text):
    path.write_text(text)
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 10))


def _wait_for(predicate, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def test_cold_start_still_builds_blocking(tmp_path, vault):
    """With no index at all there is nothing stale to serve -- must build and answer."""
    cache = str(tmp_path / "i.sqlite")
    idx = get_sqlite_index(cache, [str(vault)], serve_stale=True)
    assert [h for h in idx.retrieve("pintlers elk", 3) if h["score"] > 0]


def test_stale_index_is_served_immediately(tmp_path, vault):
    """The whole point: a changed corpus must not stall the caller."""
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    _touch(vault / "a.md", "boat maintenance and outboard motors")

    started = time.perf_counter()
    idx = get_sqlite_index(cache, [str(vault)], serve_stale=True, spawn=False)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, "serving stale must not pay rebuild cost"
    # Still answers from the OLD corpus -- stale, by design, and better than a stall.
    assert [h for h in idx.retrieve("pintlers", 3) if h["score"] > 0]


def test_stale_serve_schedules_a_rebuild(tmp_path, vault):
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    _touch(vault / "a.md", "boat maintenance and outboard motors")
    calls = []
    get_sqlite_index(cache, [str(vault)], serve_stale=True, spawn=calls.append)
    assert len(calls) == 1


def test_fresh_index_schedules_nothing(tmp_path, vault):
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    calls = []
    get_sqlite_index(cache, [str(vault)], serve_stale=True, spawn=calls.append)
    assert calls == [], "an up-to-date index must not spawn anything"


def test_serve_stale_off_by_default_rebuilds_blocking(tmp_path, vault):
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    _touch(vault / "a.md", "boat maintenance and outboard motors")
    idx = get_sqlite_index(cache, [str(vault)])
    assert [h for h in idx.retrieve("outboard motors", 3) if h["score"] > 0]
    assert [h for h in idx.retrieve("pintlers", 3) if h["score"] > 0] == []


def test_only_one_rebuild_is_scheduled_at_a_time(tmp_path, vault):
    """Every prompt sees the same stale index. Without a lock, every prompt spawns its
    own 15s rebuild and the machine melts."""
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    _touch(vault / "a.md", "boat maintenance and outboard motors")
    calls = []
    for _ in range(5):
        get_sqlite_index(cache, [str(vault)], serve_stale=True, spawn=calls.append)
    assert len(calls) == 1, f"expected one rebuild, got {len(calls)}"


def test_an_abandoned_lock_is_reclaimed(tmp_path, vault):
    """A killed rebuild must not wedge staleness forever."""
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    _touch(vault / "a.md", "boats")
    lock = cache + ".rebuild.lock"
    open(lock, "w").write("99999999")
    os.utime(lock, (0, 0))  # ancient
    calls = []
    get_sqlite_index(cache, [str(vault)], serve_stale=True, spawn=calls.append)
    assert len(calls) == 1


def test_a_corrupt_index_is_not_served_stale(tmp_path, vault):
    """Garbage has no fingerprint, so there is nothing trustworthy to serve -- rebuild."""
    cache = tmp_path / "i.sqlite"
    cache.write_bytes(b"not a database")
    idx = get_sqlite_index(str(cache), [str(vault)], serve_stale=True)
    assert [h for h in idx.retrieve("pintlers elk", 3) if h["score"] > 0]


# --- the subprocess itself -------------------------------------------------------

def test_background_rebuild_writes_nothing_to_stdout(tmp_path, vault):
    """The hook prints its JSON response to stdout. A child that writes there corrupts
    the hook's contract with Claude Code."""
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    _touch(vault / "a.md", "boat maintenance and outboard motors")

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'.');"
         "from rag_guard.sqlite_index import get_sqlite_index;"
         f"get_sqlite_index({cache!r}, [{str(vault)!r}], serve_stale=True);"
         "print('HOOK_JSON_HERE')"],
        capture_output=True, text=True,
        env={**os.environ, "RAG_GUARD_ROOTS": str(vault)})
    assert proc.stdout.strip() == "HOOK_JSON_HERE", f"child polluted stdout: {proc.stdout!r}"


def test_background_rebuild_actually_refreshes_the_index(tmp_path, vault):
    """End-to-end: spawn for real and confirm the new corpus becomes searchable."""
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    _touch(vault / "a.md", "boat maintenance and outboard motors")
    get_sqlite_index(cache, [str(vault)], serve_stale=True,
                     roots_env=str(vault))
    assert _wait_for(lambda: bool(
        [h for h in SqliteIndex(cache).retrieve("outboard motors", 3) if h["score"] > 0])), \
        "background rebuild never landed"
    assert not os.path.exists(cache + ".rebuild.lock"), "lock must be released"


def test_reindex_cli_accepts_a_backend_selection(tmp_path, vault):
    """The background path rebuilds sqlite only; rebuilding both would double the work."""
    proc = subprocess.run(
        [sys.executable, "-m", "rag_guard.reindex", "--backends", "sqlite"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd(),
             "RAG_GUARD_ROOTS": str(vault), "HOME": str(tmp_path)})
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / ".cache/rag-guard/index.sqlite").exists()
    assert not (tmp_path / ".cache/rag-guard/index.json").exists()


def test_spawned_rebuild_targets_the_given_cache_not_the_default(tmp_path, vault, monkeypatch):
    """Regression: an early version spawned `reindex` with no cache argument, so a
    background rebuild triggered from a temp vault rewrote the user's REAL index. The
    spawn must name the cache it is refreshing."""
    captured = {}

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env", {})

        class P:
            pass
        return P()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    _touch(vault / "a.md", "boats")
    get_sqlite_index(cache, [str(vault)], serve_stale=True, roots_env=str(vault))

    assert "--sqlite-cache" in captured["cmd"]
    assert cache in captured["cmd"]
    assert "--backends" in captured["cmd"] and "sqlite" in captured["cmd"]
    assert captured["env"]["RAG_GUARD_ROOTS"] == str(vault)
    assert captured["env"]["RAG_GUARD_LOCK_HELD"] == cache


def test_spawned_rebuild_is_detached_and_silent(tmp_path, vault, monkeypatch):
    """stdout is the hook's response channel; stderr noise would land in the user's
    terminal; an attached child would outlive the hook as a zombie."""
    captured = {}
    monkeypatch.setattr(subprocess, "Popen",
                        lambda cmd, **kw: captured.update(kw) or type("P", (), {})())
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    _touch(vault / "a.md", "boats")
    get_sqlite_index(cache, [str(vault)], serve_stale=True)
    assert captured["stdout"] is subprocess.DEVNULL
    assert captured["stderr"] is subprocess.DEVNULL
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["start_new_session"] is True


def test_rebuild_never_unlinks_the_index_being_served(tmp_path, vault, monkeypatch):
    """Regression for a 20s stall: reindex used to delete the cache before rebuilding,
    so a hook arriving mid-rebuild found nothing to serve stale and blocked on a rebuild
    of its own. The served file must exist continuously."""
    from rag_guard import reindex as reindex_mod
    cache = str(tmp_path / "i.sqlite")
    get_sqlite_index(cache, [str(vault)])
    assert os.path.exists(cache)

    seen = {}
    real_build_corpus = reindex_mod.build_corpus

    def spy(*a, **kw):
        seen["present_during_build"] = os.path.exists(cache)
        return real_build_corpus(*a, **kw)

    monkeypatch.setattr(reindex_mod, "build_corpus", spy)
    monkeypatch.setattr(config, "default_roots", lambda: [str(vault)])
    reindex_mod.reindex(backends=("sqlite",), sqlite_cache=cache)
    assert seen["present_during_build"] is True, "index was unlinked during rebuild"
    assert os.path.exists(cache)
