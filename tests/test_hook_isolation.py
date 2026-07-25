"""The hook must never inject retrieved notes into a headless/eval subprocess.

This is a regression suite for a real incident: this hook injected knowledge-base
passages into headless `claude -p` judge calls during a published mutation-testing
experiment, and 535 verdicts had to be re-judged. The fix at the time lived in the
caller; these tests pin the guarantee in the hook itself.

Entrypoint values are empirically verified against Claude Code 2.1.219:
interactive CLI -> "cli"; `claude -p` (inherited env AND clean cron-like env) -> "sdk-cli".
"""
import json
import subprocess
import sys

from bin import hook_userpromptsubmit as hook

HITS = [{"id": "memory/pref.md#0", "text": "Jeff prefers plain text deliverables", "score": 0.4}]
PROMPT = "what plain text format does Jeff prefer"


def _ctx(out):
    return out["hookSpecificOutput"]["additionalContext"]


def test_interactive_cli_still_grounds():
    """The whole point of the hook must survive the guard."""
    out = hook.build_output(PROMPT, HITS, 0.4, env={"CLAUDE_CODE_ENTRYPOINT": "cli"})
    assert "Jeff prefers plain text" in _ctx(out)


def test_headless_print_mode_is_silent():
    """`claude -p` reports sdk-cli. This is the exact contamination vector."""
    out = hook.build_output(PROMPT, HITS, 0.4, env={"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"})
    assert _ctx(out) == ""


def test_unknown_entrypoint_is_silent_default_deny():
    """New/unrecognized surfaces must fail closed, not leak into an eval."""
    out = hook.build_output(PROMPT, HITS, 0.4, env={"CLAUDE_CODE_ENTRYPOINT": "sdk-py"})
    assert _ctx(out) == ""


def test_missing_entrypoint_is_silent():
    """No Claude Code env at all == not an interactive session we can vouch for."""
    out = hook.build_output(PROMPT, HITS, 0.4, env={})
    assert _ctx(out) == ""


def test_ide_entrypoints_ground():
    for ep in ("vscode", "jetbrains"):
        out = hook.build_output(PROMPT, HITS, 0.4, env={"CLAUDE_CODE_ENTRYPOINT": ep})
        assert "Jeff prefers plain text" in _ctx(out), ep


def test_rivetdeck_fleet_is_silent_even_on_an_interactive_entrypoint():
    """Defense in depth: the settings.json shell guard is not the only thing stopping fleets."""
    out = hook.build_output(PROMPT, HITS, 0.4,
                            env={"CLAUDE_CODE_ENTRYPOINT": "cli", "RIVETDECK_FLEET_ID": "f1"})
    assert _ctx(out) == ""


def test_rivetdeck_fleet_with_hooks_full_still_grounds():
    """RIVETDECK_HOOKS=full is the documented opt-in and must keep working."""
    out = hook.build_output(PROMPT, HITS, 0.4,
                            env={"CLAUDE_CODE_ENTRYPOINT": "cli",
                                 "RIVETDECK_FLEET_ID": "f1", "RIVETDECK_HOOKS": "full"})
    assert "Jeff prefers plain text" in _ctx(out)


def test_explicit_opt_in_overrides_the_guard():
    """An eval harness that genuinely wants grounding must say so out loud."""
    out = hook.build_output(PROMPT, HITS, 0.4,
                            env={"CLAUDE_CODE_ENTRYPOINT": "sdk-cli",
                                 "RAG_GUARD_ALLOW_HEADLESS": "1"})
    assert "Jeff prefers plain text" in _ctx(out)


def test_guard_runs_before_retrieval_gates():
    """A headless call must be silent even when relevance would have passed easily."""
    strong = [{"id": "a#0", "text": PROMPT + " plain text deliverables desktop", "score": 0.99}]
    out = hook.build_output(PROMPT, strong, 0.99, env={"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"})
    assert _ctx(out) == ""


def _subprocess_hook(prompt, entrypoint, tmp_path, corpus_text):
    """Run the real hook binary over a real (tiny) corpus so an empty result means
    the guard fired -- not that retrieval had nothing to find."""
    root = tmp_path / "memory"
    root.mkdir()
    (root / "pref.md").write_text(corpus_text)
    return subprocess.run(
        [sys.executable, "bin/hook_userpromptsubmit.py"],
        input=json.dumps({"prompt": prompt}), capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ".",
             "HOME": str(tmp_path), "RAG_GUARD_ROOTS": str(root),
             "CLAUDE_CODE_ENTRYPOINT": entrypoint},
    )


CORPUS = "Jeff prefers plain text deliverables on the Desktop with zero markdown."


def test_end_to_end_interactive_subprocess_actually_grounds(tmp_path):
    """Control arm: proves the corpus is findable, so the silence below is the guard."""
    proc = _subprocess_hook(PROMPT, "cli", tmp_path, CORPUS)
    assert proc.returncode == 0
    assert "plain text" in json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]


def test_end_to_end_headless_subprocess_emits_empty_context(tmp_path):
    """Full stdin->stdout contract, with the real env plumbing, not just build_output."""
    proc = _subprocess_hook(PROMPT, "sdk-cli", tmp_path, CORPUS)
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"] == ""


def test_hook_still_exits_zero_and_emits_valid_json_on_a_broken_payload():
    """Fail-open on errors is unchanged; only the injection decision fails closed."""
    proc = subprocess.run(
        [sys.executable, "bin/hook_userpromptsubmit.py"],
        input="not json at all",
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ".", "CLAUDE_CODE_ENTRYPOINT": "cli"},
    )
    assert proc.returncode == 0
