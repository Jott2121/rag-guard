"""Claude Code UserPromptSubmit hook: ground the prompt in Jeff's notes.
Retrieval-only, fail-open. Injects passages + the truth-ladder protocol. Never blocks.

ISOLATION: this hook must never inject notes into a headless/eval subprocess. It once
did -- vault passages reached headless `claude -p` judge calls in a published
mutation-testing run and 535 verdicts had to be re-judged. So the *injection decision*
fails CLOSED (unrecognized surface -> stay silent) even though error handling stays
fail-open. See tests/test_hook_isolation.py."""
from __future__ import annotations

import json
import os
import sys

from rag_guard import config, service
from rag_guard.retriever import _toks

# Empirically verified against Claude Code 2.1.219: an interactive terminal session
# reports "cli"; `claude -p` reports "sdk-cli" both with an inherited env and from a
# clean cron-like env. Anything not on this list is treated as untrusted.
INTERACTIVE_ENTRYPOINTS = frozenset({"cli", "vscode", "jetbrains"})

# Escape hatch for a caller that genuinely wants grounding in a non-interactive run.
# Deliberately explicit: opting in to contamination should be a visible act.
ALLOW_HEADLESS_ENV = "RAG_GUARD_ALLOW_HEADLESS"


def is_interactive(env=None) -> bool:
    """True only for surfaces where a human is reading the answer.

    Mirrors (and backstops) the RIVETDECK_FLEET_ID shell guard in settings.json so the
    guarantee does not depend on the wiring staying correct."""
    env = os.environ if env is None else env
    if env.get(ALLOW_HEADLESS_ENV) == "1":
        return True
    if env.get("RIVETDECK_FLEET_ID") and env.get("RIVETDECK_HOOKS") != "full":
        return False
    return env.get("CLAUDE_CODE_ENTRYPOINT", "") in INTERACTIVE_ENTRYPOINTS

_PROTOCOL = (
    "GROUNDING PROTOCOL: Prefer the notes above. If they don't cover the question and it "
    "may be newer than your training cutoff, search the web, corroborate the claim across "
    ">=2 independent sources (prefer primary/official over social), cite them, and flag "
    "conflicts or anything contradicting Jeff's notes. State the answer's confidence: "
    "grounded / web-verified / single-source / unverified."
)


def _top_overlap(prompt, hits):
    if not hits:
        return 0
    return len(set(_toks(prompt)) & set(_toks(hits[0]["text"])))


def build_output(prompt, hits, support, env=None):
    if not is_interactive(env) or not hits or support < config.MIN_SCORE or \
            _top_overlap(prompt, hits) < config.HOOK_MIN_OVERLAP:
        ctx = ""
    else:
        passages = "\n".join(f"- ({h['id']}) {h['text']}" for h in hits)
        ctx = f"Relevant notes from Jeff's knowledge base:\n{passages}\n\n{_PROTOCOL}"
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ctx}}


def main():
    try:
        payload = json.load(sys.stdin)
        if not is_interactive():
            # Short-circuit: skip retrieval entirely rather than retrieve-then-discard.
            print(json.dumps(build_output("", [], 0.0, env={})))
            sys.exit(0)
        hits = service.query(payload.get("prompt", ""), 5)
        support = max((h["score"] for h in hits), default=0.0)
        print(json.dumps(build_output(payload.get("prompt", ""), hits, support)))
    except Exception:
        pass  # fail-open
    sys.exit(0)


if __name__ == "__main__":
    main()
