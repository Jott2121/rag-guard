"""What the hook records, end-to-end through the real binary.

The point of this log is to replace intuition about the gates with data, so the cases
that matter most are the SILENT ones -- they are the ones with no other trace.
"""
import json
import subprocess
import sys

CORPUS = ("Jeff prefers plain text deliverables on the Desktop with zero markdown. "
          "The Sabot mutation score work shipped with a strict scorer.")


def _run(prompt, entrypoint, tmp_path, *, log=True, roots=None):
    root = tmp_path / "memory"
    if not root.exists():
        root.mkdir()
        (root / "pref.md").write_text(CORPUS)
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": ".", "HOME": str(tmp_path),
           "RAG_GUARD_ROOTS": roots or str(root), "CLAUDE_CODE_ENTRYPOINT": entrypoint}
    if log:
        env["RAG_GUARD_HOOK_LOG"] = "1"
    proc = subprocess.run(
        [sys.executable, "bin/hook_userpromptsubmit.py"],
        input=json.dumps({"prompt": prompt, "session_id": "sess-1", "cwd": "/tmp"}),
        capture_output=True, text=True, env=env)
    logfile = tmp_path / ".local/state/rag-guard/hook-fires.jsonl"
    events = [json.loads(x) for x in logfile.read_text().splitlines()] if logfile.exists() else []
    return json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"], events


def test_a_fire_is_recorded_with_resolvable_chunk_paths(tmp_path):
    ctx, events = _run("what plain text format does Jeff prefer", "cli", tmp_path)
    assert ctx, "precondition: this prompt should ground"
    (ev,) = events
    assert ev["fired"] is True and ev["reason"] == "fired"
    assert ev["session_id"] == "sess-1"
    assert ev["chunks"], "a fire with no recorded chunks is useless for the analysis"
    for c in ev["chunks"]:
        assert c["path"] and c["path"].endswith("pref.md"), c
    assert ev["prompt"].startswith("what plain text format")


def test_a_silent_decision_is_recorded_with_its_reason(tmp_path):
    """Underfiring is the invisible failure -- the session just proceeds ungrounded."""
    ctx, events = _run("quantum chromodynamics lattice gauge", "cli", tmp_path)
    assert ctx == ""
    (ev,) = events
    assert ev["fired"] is False
    assert ev["reason"] in {"no_hits", "below_min_score", "below_min_overlap"}
    assert ev["chunks"] == []
    assert ev["prompt"], "we need the prompt to judge whether silence was correct"


def test_a_suppressed_headless_call_is_recorded_without_the_prompt(tmp_path):
    """Verifies the guard fires in the wild, without archiving eval/judge payloads."""
    ctx, events = _run("what plain text format does Jeff prefer", "sdk-cli", tmp_path)
    assert ctx == ""
    (ev,) = events
    assert ev["fired"] is False and ev["reason"] == "not_interactive"
    assert ev["prompt"] is None
    assert ev["entrypoint"] == "sdk-cli"


def test_logging_is_off_unless_opted_in(tmp_path):
    ctx, events = _run("what plain text format does Jeff prefer", "cli", tmp_path, log=False)
    assert ctx, "grounding must be unaffected by the logging switch"
    assert events == []


def test_grounding_survives_an_unwritable_log(tmp_path):
    """Instrumentation must never be able to break the thing it instruments."""
    root = tmp_path / "memory"
    root.mkdir()
    (root / "pref.md").write_text(CORPUS)
    state = tmp_path / ".local/state/rag-guard"
    state.mkdir(parents=True)
    state.chmod(0o500)
    try:
        proc = subprocess.run(
            [sys.executable, "bin/hook_userpromptsubmit.py"],
            input=json.dumps({"prompt": "what plain text format does Jeff prefer"}),
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ".", "HOME": str(tmp_path),
                 "RAG_GUARD_ROOTS": str(root), "CLAUDE_CODE_ENTRYPOINT": "cli",
                 "RAG_GUARD_HOOK_LOG": "1"})
        assert proc.returncode == 0
        assert "plain text" in json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
    finally:
        state.chmod(0o700)
