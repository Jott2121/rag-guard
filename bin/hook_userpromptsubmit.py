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
import time

from rag_guard import config, hooklog, service
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

# Softened deliberately. Retrieval is keyword-based and fires on ~81% of prompts; measured
# relevance is 38.5%, so roughly three in five injections do not cover the question. Telling
# the model to "prefer the notes above" unconditionally is an instruction to trust
# possibly-irrelevant context. The ids are vault-qualified so a passage can be traced, and
# the index may be seconds-to-minutes behind on purpose (serve-stale), so both caveats are
# stated rather than implied.
_PROTOCOL = (
    "GROUNDING PROTOCOL: The notes above were keyword-retrieved and may be unrelated, "
    "partial, or slightly out of date. Check that a passage actually addresses the question "
    "before relying on it, and say so if none do. Where they do apply, prefer them over "
    "recollection and cite the note id (ids are vault-qualified, e.g. Hunt-621-Wiki/Note.md). "
    "If the question isn't covered and may postdate your training, search the web, "
    "corroborate across >=2 independent sources (prefer primary/official over social), cite "
    "them, and flag conflicts with the notes. State the answer's confidence: "
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


def decide(prompt, hits, support):
    """Why the hook did what it did — the shape the log needs, and the reason build_output
    can stay a pure formatter."""
    if not hits:
        return False, "no_hits"
    if support < config.MIN_SCORE:
        return False, "below_min_score"
    if _top_overlap(prompt, hits) < config.HOOK_MIN_OVERLAP:
        return False, "below_min_overlap"
    return True, "fired"


def _log(payload, *, fired, reason, hits=(), support=0.0, elapsed_ms=0, prompt=None):
    """Record what happened. Silent decisions are logged too: underfiring leaves no other
    trace, and 'the session just proceeded ungrounded' is invisible without this."""
    hooklog.log_event({
        "session_id": payload.get("session_id"),
        "cwd": payload.get("cwd"),
        "entrypoint": os.environ.get("CLAUDE_CODE_ENTRYPOINT"),
        # Suppressed headless calls log NO prompt: those are the eval/judge payloads the
        # isolation guard exists to keep at arm's length.
        "prompt": (prompt or "")[:500] if prompt is not None else None,
        "fired": fired,
        "reason": reason,
        "support": round(support, 6),
        "elapsed_ms": elapsed_ms,
        "chunks": [{"id": h["id"], "path": _resolve(h["id"]), "score": h["score"],
                    "chars": len(h["text"])} for h in hits],
    }, path=config.hook_log_path())


def _resolve(chunk_id):
    """Chunk id -> absolute path. Ids are vault-qualified and relative to the PARENT of
    each root (see corpus._add_file), so join against that same base."""
    rel = chunk_id.split("#")[0]
    for root in config.default_roots():
        root_dir = root if os.path.isdir(root) else os.path.dirname(root)
        candidate = os.path.join(os.path.dirname(root_dir), rel)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    payload = {}
    try:
        payload = json.load(sys.stdin)
        if not is_interactive():
            # Short-circuit: skip retrieval entirely rather than retrieve-then-discard.
            print(json.dumps(build_output("", [], 0.0, env={})))
            _log(payload, fired=False, reason="not_interactive", prompt=None)
            sys.exit(0)
        prompt = payload.get("prompt", "")
        started = time.perf_counter()
        hits = service.query(prompt, 5)
        support = max((h["score"] for h in hits), default=0.0)
        print(json.dumps(build_output(prompt, hits, support)))
        fired, reason = decide(prompt, hits, support)
        _log(payload, fired=fired, reason=reason, hits=hits if fired else [],
             support=support, prompt=prompt,
             elapsed_ms=int((time.perf_counter() - started) * 1000))
    except Exception:
        pass  # fail-open
    sys.exit(0)


if __name__ == "__main__":
    main()
