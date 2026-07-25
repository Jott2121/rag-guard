from bin import hook_userpromptsubmit as hook

# The hook only grounds on an interactive surface (see test_hook_isolation.py). These
# relevance-gate tests pass it explicitly so they never depend on the ambient env --
# without this they pass on Jeff's machine and fail in CI.
INTERACTIVE = {"CLAUDE_CODE_ENTRYPOINT": "cli"}


def test_output_contract_when_grounded():
    hits = [{"id": "memory/pref.md#0", "text": "Jeff prefers plain text deliverables", "score": 0.4}]
    out = hook.build_output("what plain text format does Jeff prefer", hits, 0.4, env=INTERACTIVE)
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "Jeff prefers plain text" in ctx and "corroborate" in ctx.lower()


def test_silent_when_low_support():
    out = hook.build_output("obscure", [], 0.0, env=INTERACTIVE)
    assert out["hookSpecificOutput"]["additionalContext"] == ""


def test_silent_when_only_one_word_overlaps():
    # spurious single-word match must NOT ground (needs >=2 shared content words)
    hits = [{"id": "x#0", "text": "weather patterns in the pintlers during elk season", "score": 0.4}]
    out = hook.build_output("what is the weather today", hits, 0.4, env=INTERACTIVE)
    assert out["hookSpecificOutput"]["additionalContext"] == ""
