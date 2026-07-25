"""The fire-to-transcript join. If this is wrong, the push-vs-pull decision is wrong."""
import io
import json

from bin import hook_report


def _transcript(tmp_path, session_id, entries):
    p = tmp_path / f"{session_id}.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries))
    return p


def _fire(session_id, ts, paths):
    return {"ts": ts, "session_id": session_id, "fired": True, "reason": "fired",
            "chunks": [{"id": f"{i}", "path": p, "score": 0.3, "chars": 800}
                       for i, p in enumerate(paths)]}


def test_counts_a_touch_that_follows_the_fire(tmp_path):
    _transcript(tmp_path, "s1", [
        {"timestamp": "2026-07-24T12:00:10Z",
         "message": {"content": [{"type": "tool_use", "name": "Read",
                                  "input": {"file_path": "/vault/a.md"}}]}},
    ])
    ts = hook_report._iso_to_epoch("2026-07-24T12:00:00Z")
    rows = hook_report.analyze([_fire("s1", ts, ["/vault/a.md"])], str(tmp_path))
    assert len(rows) == 1 and rows[0]["used"] == {"/vault/a.md"}


def test_ignores_a_touch_that_precedes_the_fire(tmp_path):
    """A file read BEFORE we injected it is not evidence the injection did anything."""
    _transcript(tmp_path, "s1", [
        {"timestamp": "2026-07-24T11:59:00Z",
         "message": {"content": [{"type": "tool_use", "name": "Read",
                                  "input": {"file_path": "/vault/a.md"}}]}},
    ])
    ts = hook_report._iso_to_epoch("2026-07-24T12:00:00Z")
    rows = hook_report.analyze([_fire("s1", ts, ["/vault/a.md"])], str(tmp_path))
    assert rows[0]["used"] == set()


def test_ignores_touches_of_files_we_did_not_inject(tmp_path):
    _transcript(tmp_path, "s1", [
        {"timestamp": "2026-07-24T12:00:10Z",
         "message": {"content": [{"type": "tool_use", "name": "Read",
                                  "input": {"file_path": "/vault/other.md"}}]}},
    ])
    ts = hook_report._iso_to_epoch("2026-07-24T12:00:00Z")
    rows = hook_report.analyze([_fire("s1", ts, ["/vault/a.md"])], str(tmp_path))
    assert rows[0]["used"] == set()


def test_finds_a_path_embedded_in_a_bash_command(tmp_path):
    """Not every touch is a Read with a file_path -- grep/cat via Bash counts too."""
    _transcript(tmp_path, "s1", [
        {"timestamp": "2026-07-24T12:00:10Z",
         "message": {"content": [{"type": "tool_use", "name": "Bash",
                                  "input": {"command": "grep -n x \"/vault/a.md\""}}]}},
    ])
    ts = hook_report._iso_to_epoch("2026-07-24T12:00:00Z")
    rows = hook_report.analyze([_fire("s1", ts, ["/vault/a.md"])], str(tmp_path))
    assert rows[0]["used"] == {"/vault/a.md"}


def test_missing_transcript_is_dropped_not_counted_as_unused(tmp_path):
    """A session with no transcript is missing data, not evidence of non-use --
    counting it as 'never touched' would bias the headline number downward."""
    ts = hook_report._iso_to_epoch("2026-07-24T12:00:00Z")
    rows = hook_report.analyze([_fire("absent", ts, ["/vault/a.md"])], str(tmp_path))
    assert rows == [] or rows[0]["used"] == set()
    out = io.StringIO()
    hook_report.report([_fire("absent", ts, ["/vault/a.md"])], str(tmp_path), out=out)
    assert "USAGE JOIN" in out.getvalue()


def test_silent_and_suppressed_decisions_are_excluded_from_the_join(tmp_path):
    events = [{"ts": 1, "session_id": "s1", "fired": False, "reason": "below_min_overlap"},
              {"ts": 1, "session_id": "s1", "fired": False, "reason": "not_interactive"}]
    assert hook_report.analyze(events, str(tmp_path)) == []


def test_report_summarises_fire_rate_over_interactive_decisions_only(tmp_path):
    """Suppressed headless calls must not dilute the fire rate -- they were never
    candidates for grounding in the first place."""
    ts = hook_report._iso_to_epoch("2026-07-24T12:00:00Z")
    events = [_fire("s1", ts, ["/vault/a.md"]),
              {"ts": ts, "session_id": "s1", "fired": False, "reason": "below_min_overlap"},
              {"ts": ts, "session_id": "s2", "fired": False, "reason": "not_interactive"}]
    out = io.StringIO()
    hook_report.report(events, str(tmp_path), out=out)
    text = out.getvalue()
    assert "interactive decisions         : 2" in text
    assert "50.0%" in text


def test_report_on_an_empty_log_explains_itself(tmp_path):
    out = io.StringIO()
    assert hook_report.report([], str(tmp_path), out=out) == 1
    assert "RAG_GUARD_HOOK_LOG" in out.getvalue()
