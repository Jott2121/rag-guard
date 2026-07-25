"""A background rebuild that keeps failing must not become unbounded silent staleness.

Serving stale is a deliberate trade. It stops being defensible the moment nobody can tell
how stale, or that rebuilds have stopped succeeding at all. One unparseable note or a full
disk kills the detached child every time: the lock releases, the next prompt serves stale
again, and this repeats forever. Routing the child's stderr to DEVNULL was right for
stdout hygiene and also deleted the only failure signal, so the signal has to be a file.
"""
import json
import os
import time

from rag_guard import rebuild_health as rh


def test_no_marker_means_unknown_not_healthy(tmp_path):
    """Absence of evidence is not evidence of freshness."""
    status = rh.read_status(str(tmp_path / "i.sqlite"))
    assert status["state"] == "unknown"
    assert status["stale_seconds"] is None


def test_success_is_recorded(tmp_path):
    cache = str(tmp_path / "i.sqlite")
    rh.record_success(cache, chunks=1234)
    status = rh.read_status(cache)
    assert status["state"] == "ok" and status["chunks"] == 1234
    assert status["consecutive_failures"] == 0


def test_failure_is_recorded_with_its_reason(tmp_path):
    cache = str(tmp_path / "i.sqlite")
    rh.record_failure(cache, "MemoryError: out of memory")
    status = rh.read_status(cache)
    assert status["state"] == "failed"
    assert "MemoryError" in status["error"]
    assert status["consecutive_failures"] == 1


def test_consecutive_failures_accumulate(tmp_path):
    cache = str(tmp_path / "i.sqlite")
    for _ in range(3):
        rh.record_failure(cache, "boom")
    assert rh.read_status(cache)["consecutive_failures"] == 3


def test_a_success_clears_the_failure_streak(tmp_path):
    cache = str(tmp_path / "i.sqlite")
    rh.record_failure(cache, "boom")
    rh.record_failure(cache, "boom")
    rh.record_success(cache, chunks=10)
    assert rh.read_status(cache)["consecutive_failures"] == 0


def test_staleness_age_is_reported(tmp_path):
    cache = str(tmp_path / "i.sqlite")
    rh.record_success(cache, chunks=10)
    marker = rh.marker_path(cache)
    payload = json.loads(open(marker).read())
    payload["at"] = time.time() - 7200
    open(marker, "w").write(json.dumps(payload))
    assert rh.read_status(cache)["stale_seconds"] >= 7000


def test_warning_only_fires_when_rebuilds_are_actually_broken(tmp_path):
    """A healthy system must stay silent -- a warning on every prompt is noise, and noise
    is how a real warning gets ignored."""
    cache = str(tmp_path / "i.sqlite")
    rh.record_success(cache, chunks=10)
    assert rh.warning(cache) is None


def test_warning_fires_after_repeated_failures(tmp_path):
    cache = str(tmp_path / "i.sqlite")
    for _ in range(rh.FAILURE_THRESHOLD):
        rh.record_failure(cache, "boom")
    text = rh.warning(cache)
    assert text and "rebuild" in text.lower()


def test_warning_fires_when_a_stale_index_has_not_refreshed_in_ages(tmp_path):
    cache = str(tmp_path / "i.sqlite")
    rh.record_success(cache, chunks=10)
    marker = rh.marker_path(cache)
    payload = json.loads(open(marker).read())
    payload["at"] = time.time() - (rh.STALE_WARN_SECONDS + 60)
    open(marker, "w").write(json.dumps(payload))
    assert rh.warning(cache) is not None


def test_corrupt_marker_degrades_to_unknown(tmp_path):
    cache = str(tmp_path / "i.sqlite")
    open(rh.marker_path(cache), "w").write("{ not json")
    assert rh.read_status(cache)["state"] == "unknown"
    assert rh.warning(cache) is None


def test_recording_never_raises_on_an_unwritable_location(tmp_path):
    blocked = tmp_path / "ro"
    blocked.mkdir()
    os.chmod(blocked, 0o500)
    try:
        rh.record_success(str(blocked / "i.sqlite"), chunks=1)
        rh.record_failure(str(blocked / "i.sqlite"), "boom")
    finally:
        os.chmod(blocked, 0o700)


def test_reindex_records_success(tmp_path, monkeypatch):
    from rag_guard import config, reindex as reindex_mod
    root = tmp_path / "memory"
    root.mkdir()
    (root / "n.md").write_text("elk hunting in the pintlers")
    cache = str(tmp_path / "i.sqlite")
    monkeypatch.setattr(config, "default_roots", lambda: [str(root)])
    reindex_mod.main(["--backends", "sqlite", "--sqlite-cache", cache])
    assert rh.read_status(cache)["state"] == "ok"


def test_reindex_records_failure_and_still_exits_nonzero(tmp_path, monkeypatch):
    """The child dies silently by design. If it does not write WHY, nobody ever learns."""
    from rag_guard import config, reindex as reindex_mod
    root = tmp_path / "memory"
    root.mkdir()
    (root / "n.md").write_text("elk")
    cache = str(tmp_path / "i.sqlite")
    monkeypatch.setattr(config, "default_roots", lambda: [str(root)])

    def boom(*a, **kw):
        raise MemoryError("simulated")

    monkeypatch.setattr(reindex_mod, "build_corpus", boom)
    rc = reindex_mod.main(["--backends", "sqlite", "--sqlite-cache", cache])
    assert rc != 0
    status = rh.read_status(cache)
    assert status["state"] == "failed" and "simulated" in status["error"]


def test_hook_appends_the_warning_only_when_rebuilds_are_broken(tmp_path, monkeypatch):
    """End-to-end: the signal has to reach the prompt, or the telemetry is a diary."""
    from bin import hook_userpromptsubmit as hook
    from rag_guard import config

    cache = str(tmp_path / "i.sqlite")
    monkeypatch.setattr(config, "sqlite_cache_path", lambda: cache)
    hits = [{"id": "memory/pref.md#0", "text": "Jeff prefers plain text", "score": 0.4}]
    prompt = "what plain text format does Jeff prefer"
    env = {"CLAUDE_CODE_ENTRYPOINT": "cli"}

    rh.record_success(cache, chunks=10)
    healthy = hook.build_output(prompt, hits, 0.4, env=env)
    assert "out of date" not in healthy["hookSpecificOutput"]["additionalContext"]

    for _ in range(rh.FAILURE_THRESHOLD):
        rh.record_failure(cache, "MemoryError: simulated")
    broken = hook.build_output(prompt, hits, 0.4, env=env)
    ctx = broken["hookSpecificOutput"]["additionalContext"]
    assert "out of date" in ctx and "MemoryError" in ctx
    assert "Jeff prefers plain text" in ctx, "grounding must survive the warning"
