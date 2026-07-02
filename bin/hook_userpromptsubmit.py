"""Claude Code UserPromptSubmit hook: ground the prompt in Jeff's notes.
Retrieval-only, fail-open. Injects passages + the truth-ladder protocol. Never blocks."""
from __future__ import annotations

import json
import sys

from rag_guard import config, service

_PROTOCOL = (
    "GROUNDING PROTOCOL: Prefer the notes above. If they don't cover the question and it "
    "may be newer than your training cutoff, search the web, corroborate the claim across "
    ">=2 independent sources (prefer primary/official over social), cite them, and flag "
    "conflicts or anything contradicting Jeff's notes. State the answer's confidence: "
    "grounded / web-verified / single-source / unverified."
)


def build_output(prompt, hits, support):
    if support < config.MIN_SCORE or not hits:
        ctx = ""
    else:
        passages = "\n".join(f"- ({h['id']}) {h['text']}" for h in hits)
        ctx = f"Relevant notes from Jeff's knowledge base:\n{passages}\n\n{_PROTOCOL}"
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ctx}}


def main():
    try:
        payload = json.load(sys.stdin)
        hits = service.query(payload.get("prompt", ""), 5)
        support = max((h["score"] for h in hits), default=0.0)
        print(json.dumps(build_output(payload.get("prompt", ""), hits, support)))
    except Exception:
        pass  # fail-open
    sys.exit(0)


if __name__ == "__main__":
    main()
